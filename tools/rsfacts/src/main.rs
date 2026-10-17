// rsfacts: the Rust syntax-fact extractor behind SNDL's Rust-aware gates (plan section 10, R0.2).
//
// Parses each .rs file with syn (pure Rust; chosen over tree-sitter-rust on measurement, see
// src/snn_data_lake/gates/rust_facts.py) and prints ONE JSON line per file. It reports syntax
// only: name resolution (`crate::`/`super::`/`self::`), the PyO3 export map and every policy
// rule live in the Python gates. Macro bodies are token streams to syn, so a call written inside
// `vec![..]`, `azip!(..)` or `s![..]` is NOT a call here; its raw tokens are kept in `macros` so
// a gate can disclose (or regex) them. Paths come from argv, or from stdin when argv is `-`.
use quote::ToTokens;
use serde_json::{json, Value};
use std::io::BufRead;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

const SCHEMA: u32 = 2;

#[derive(Default)]
struct Facts {
    fns: Vec<Value>,
    types: Vec<Value>,
    mods: Vec<Value>,
    uses: Vec<Value>,
    consts: Vec<Value>,
    calls: Vec<Value>,
    mcalls: Vec<Value>,
    macros: Vec<Value>,
    index: Vec<Value>,
    assigns: Vec<Value>,
    lets: Vec<Value>,
    fors: Vec<Value>,
    path_roots: Vec<String>,
    paths: Vec<String>,
    test_paths: Vec<String>,
    // context
    cur_fn: Vec<String>,
    // index into `fns` of the innermost enclosing fn: exact attribution where names and lines collide
    cur_fn_id: Vec<usize>,
    cur_impl: Vec<(String, Option<String>, Vec<String>)>,
    cur_mod: Vec<String>,
    cfg_test: usize,
    // enclosing `{ .. }` ordinals (file-unique), outermost first; Python treats a let as visible
    // where its scope is a prefix of the use's
    scope: Vec<usize>,
    blocks: usize,
}

fn line<S: Spanned>(s: &S) -> usize {
    s.span().start().line
}
// Column orders events that share a line (e.g. `let d = ..; w += &d; let d = ..;` on one line).
fn col<S: Spanned>(s: &S) -> usize {
    s.span().start().column
}
fn txt<T: ToTokens>(t: &T) -> String {
    t.to_token_stream().to_string()
}
fn attrs_of(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| !a.path().is_ident("doc"))
        .map(|a| {
            let p = txt(a.path()).replace(' ', "");
            match &a.meta {
                syn::Meta::List(l) => format!("{p}({})", l.tokens),
                syn::Meta::NameValue(nv) => format!("{p} = {}", txt(&nv.value)),
                syn::Meta::Path(_) => p,
            }
        })
        .collect()
}
fn docs_of(attrs: &[syn::Attribute]) -> String {
    let mut out = Vec::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let syn::Meta::NameValue(nv) = &a.meta {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
            {
                out.push(s.value());
            }
        }
    }
    out.join("\n")
}
// Value of a cfg predicate in a NON-test build, three-valued: Some(false) = never compiled outside
// tests, Some(true) = always compiled, None = depends on other cfg atoms (features, targets).
fn cfg_outside_tests(m: &syn::Meta) -> Option<bool> {
    let list = |l: &syn::MetaList| -> Vec<syn::Meta> {
        l.parse_args_with(
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
        )
        .map(|p| p.into_iter().collect())
        .unwrap_or_default()
    };
    match m {
        syn::Meta::Path(p) if p.is_ident("test") => Some(false),
        syn::Meta::List(l) if l.path.is_ident("all") => {
            let v: Vec<_> = list(l).iter().map(cfg_outside_tests).collect();
            if v.contains(&Some(false)) {
                Some(false)
            } else if v.iter().all(|x| *x == Some(true)) {
                Some(true)
            } else {
                None
            }
        }
        syn::Meta::List(l) if l.path.is_ident("any") => {
            let v: Vec<_> = list(l).iter().map(cfg_outside_tests).collect();
            if v.contains(&Some(true)) {
                Some(true)
            } else if v.iter().all(|x| *x == Some(false)) {
                Some(false)
            } else {
                None
            }
        }
        syn::Meta::List(l) if l.path.is_ident("not") => {
            list(l).first().and_then(cfg_outside_tests).map(|b| !b)
        }
        _ => None,
    }
}
// Test-only code: a test fn (`#[test]`, `#[tokio::test]`, ...) or a `#[cfg(..)]` whose predicate is
// FALSE in every non-test build: `cfg(test)`, `cfg(all(test, ..))`. `cfg(not(test))`,
// `cfg(any(test, feature = ..))` and `cfg(not(any(test, ..)))` are production code.
fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        let last = a.path().segments.last().map(|s| s.ident.to_string());
        if last.as_deref() == Some("test") {
            return true;
        }
        if !a.path().is_ident("cfg") {
            return false;
        }
        match &a.meta {
            syn::Meta::List(l) => {
                l.parse_args::<syn::Meta>()
                    .ok()
                    .and_then(|m| cfg_outside_tests(&m))
                    == Some(false)
            }
            _ => false,
        }
    })
}
fn vis_of(v: &syn::Visibility) -> String {
    txt(v).replace(' ', "")
}

impl Facts {
    fn ctx(&self) -> (Option<String>, String, bool) {
        (
            self.cur_fn.last().cloned(),
            self.cur_mod.join("::"),
            self.cfg_test > 0,
        )
    }
    fn fn_rec(
        &mut self,
        sig: &syn::Signature,
        vis: &syn::Visibility,
        attrs: &[syn::Attribute],
        has_body: bool,
    ) {
        let (imp, tr, iattrs) = match self.cur_impl.last() {
            Some((a, b, c)) => (Some(a.clone()), b.clone(), c.clone()),
            None => (None, None, Vec::new()),
        };
        let params: Vec<String> = sig.inputs.iter().map(txt).collect();
        let (parent, modp, cfg) = self.ctx();
        let id = self.fns.len();
        self.fns.push(json!({"id": id, "name": sig.ident.to_string(), "line": line(&sig.ident),
            "vis": vis_of(vis), "impl": imp, "trait": tr, "impl_attrs": iattrs, "attrs": attrs_of(attrs),
            "doc": docs_of(attrs), "params": params, "ret": txt(&sig.output), "has_body": has_body,
            "cfg_test": cfg || is_cfg_test(attrs), "parent": parent, "mod": modp}));
    }
    fn type_rec(
        &mut self,
        kind: &str,
        ident: &syn::Ident,
        vis: &syn::Visibility,
        attrs: &[syn::Attribute],
    ) {
        let (_, modp, cfg) = self.ctx();
        self.types.push(json!({"kind": kind, "name": ident.to_string(), "line": line(ident),
            "vis": vis_of(vis), "attrs": attrs_of(attrs), "cfg_test": cfg || is_cfg_test(attrs), "mod": modp}));
    }
    fn flatten_use(
        &mut self,
        prefix: &str,
        t: &syn::UseTree,
        ln: usize,
        attrs: &[String],
        vis: &str,
    ) {
        let (_, modp, cfg) = self.ctx();
        let mut push = |path: String, alias: Option<String>| {
            self.uses.push(
                json!({"path": path, "alias": alias, "line": ln, "attrs": attrs, "vis": vis,
                "cfg_test": cfg, "mod": modp}),
            );
        };
        match t {
            syn::UseTree::Path(p) => {
                let pre = format!("{prefix}{}::", p.ident);
                self.flatten_use(&pre, &p.tree, ln, attrs, vis)
            }
            syn::UseTree::Name(n) => push(format!("{prefix}{}", n.ident), None),
            syn::UseTree::Rename(r) => {
                push(format!("{prefix}{}", r.ident), Some(r.rename.to_string()))
            }
            syn::UseTree::Glob(_) => push(format!("{prefix}*"), None),
            syn::UseTree::Group(g) => {
                for i in &g.items {
                    self.flatten_use(prefix, i, ln, attrs, vis)
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for Facts {
    fn visit_block(&mut self, i: &'ast syn::Block) {
        self.blocks += 1;
        self.scope.push(self.blocks);
        visit::visit_block(self, i);
        self.scope.pop();
    }
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.fn_rec(&i.sig, &i.vis, &i.attrs, true);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        self.cur_fn.push(i.sig.ident.to_string());
        self.cur_fn_id.push(self.fns.len() - 1);
        visit::visit_item_fn(self, i);
        self.cur_fn.pop();
        self.cur_fn_id.pop();
        self.cfg_test -= t;
    }
    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.fn_rec(&i.sig, &i.vis, &i.attrs, true);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        self.cur_fn.push(i.sig.ident.to_string());
        self.cur_fn_id.push(self.fns.len() - 1);
        visit::visit_impl_item_fn(self, i);
        self.cur_fn.pop();
        self.cur_fn_id.pop();
        self.cfg_test -= t;
    }
    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        self.fn_rec(
            &i.sig,
            &syn::Visibility::Inherited,
            &i.attrs,
            i.default.is_some(),
        );
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        self.cur_fn.push(i.sig.ident.to_string());
        self.cur_fn_id.push(self.fns.len() - 1);
        visit::visit_trait_item_fn(self, i);
        self.cur_fn.pop();
        self.cur_fn_id.pop();
        self.cfg_test -= t;
    }
    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let tr = i.trait_.as_ref().map(|(p, _)| txt(p).replace(' ', ""));
        self.cur_impl
            .push((txt(&*i.self_ty).replace(' ', ""), tr, attrs_of(&i.attrs)));
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_impl(self, i);
        self.cfg_test -= t;
        self.cur_impl.pop();
    }
    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.type_rec("trait", &i.ident, &i.vis, &i.attrs);
        self.cur_impl.push((
            i.ident.to_string(),
            Some(i.ident.to_string()),
            attrs_of(&i.attrs),
        ));
        visit::visit_item_trait(self, i);
        self.cur_impl.pop();
    }
    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.type_rec("struct", &i.ident, &i.vis, &i.attrs);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_struct(self, i);
        self.cfg_test -= t;
    }
    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.type_rec("enum", &i.ident, &i.vis, &i.attrs);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_enum(self, i);
        self.cfg_test -= t;
    }
    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        self.type_rec("union", &i.ident, &i.vis, &i.attrs);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_union(self, i);
        self.cfg_test -= t;
    }
    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.type_rec("type", &i.ident, &i.vis, &i.attrs);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_type(self, i);
        self.cfg_test -= t;
    }
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let (_, modp, cfg) = self.ctx();
        self.mods.push(
            json!({"name": i.ident.to_string(), "line": line(&i.ident), "vis": vis_of(&i.vis),
            "inline": i.content.is_some(), "attrs": attrs_of(&i.attrs),
            "cfg_test": cfg || is_cfg_test(&i.attrs), "mod": modp}),
        );
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        self.cur_mod.push(i.ident.to_string());
        visit::visit_item_mod(self, i);
        self.cur_mod.pop();
        self.cfg_test -= t;
    }
    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let attrs = attrs_of(&i.attrs);
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        self.flatten_use("", &i.tree, line(i), &attrs, &vis_of(&i.vis));
        self.cfg_test -= t;
    }
    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        self.path_roots.push(i.ident.to_string());
        visit::visit_item_extern_crate(self, i);
    }
    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        let (_, modp, cfg) = self.ctx();
        self.consts.push(json!({"kind": "const", "name": i.ident.to_string(), "line": line(&i.ident),
            "ty": txt(&*i.ty), "expr": txt(&*i.expr), "attrs": attrs_of(&i.attrs), "cfg_test": cfg || is_cfg_test(&i.attrs), "mod": modp}));
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_const(self, i);
        self.cfg_test -= t;
    }
    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        let (_, modp, cfg) = self.ctx();
        self.consts.push(json!({"kind": "static", "name": i.ident.to_string(), "line": line(&i.ident),
            "ty": txt(&*i.ty), "expr": txt(&*i.expr), "attrs": attrs_of(&i.attrs), "cfg_test": cfg || is_cfg_test(&i.attrs), "mod": modp}));
        let t = is_cfg_test(&i.attrs) as usize;
        self.cfg_test += t;
        visit::visit_item_static(self, i);
        self.cfg_test -= t;
    }
    fn visit_path(&mut self, p: &'ast syn::Path) {
        if p.segments.len() >= 2 {
            self.path_roots.push(p.segments[0].ident.to_string());
            let full: Vec<String> = p.segments.iter().map(|s| s.ident.to_string()).collect();
            // production paths only: a test-scoped path must never make a module reachable
            if self.cfg_test == 0 {
                self.paths.push(full.join("::"));
            } else {
                self.test_paths.push(full.join("::"));
            }
        }
        visit::visit_path(self, p);
    }
    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*i.func {
            let (f, _, cfg) = self.ctx();
            self.calls.push(
                json!({"path": txt(&p.path).replace(' ', ""), "line": line(i), "col": col(i), "fn": f, "fn_id": self.cur_fn_id.last(),
                "scope": self.scope, "args": i.args.iter().map(txt).collect::<Vec<_>>(), "cfg_test": cfg}),
            );
        }
        visit::visit_expr_call(self, i);
    }
    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let (f, _, cfg) = self.ctx();
        self.mcalls.push(
            json!({"name": i.method.to_string(), "line": line(&i.method), "col": col(&i.method), "fn": f, "fn_id": self.cur_fn_id.last(),
            "scope": self.scope, "recv": txt(&*i.receiver), "args": i.args.iter().map(txt).collect::<Vec<_>>(),
            "turbofish": i.turbofish.as_ref().map(txt), "cfg_test": cfg}),
        );
        visit::visit_expr_method_call(self, i);
    }
    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        let (f, _, cfg) = self.ctx();
        self.index.push(
            json!({"expr": txt(&*i.expr), "index": txt(&*i.index), "line": line(i), "col": col(i), "fn": f, "fn_id": self.cur_fn_id.last(),
            "cfg_test": cfg}),
        );
        visit::visit_expr_index(self, i);
    }
    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        let (f, _, cfg) = self.ctx();
        self.assigns.push(
            json!({"lhs": txt(&*i.left), "op": "=", "rhs": txt(&*i.right), "line": line(i), "col": col(i),
            "end_line": i.span().end().line, "end_col": i.span().end().column,
            "fn": f, "fn_id": self.cur_fn_id.last(), "cfg_test": cfg}),
        );
        visit::visit_expr_assign(self, i);
    }
    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        let op = txt(&i.op);
        if op.ends_with('=') && !matches!(op.as_str(), "==" | "!=" | "<=" | ">=") {
            let (f, _, cfg) = self.ctx();
            self.assigns.push(
                json!({"lhs": txt(&*i.left), "op": op, "rhs": txt(&*i.right), "line": line(i), "col": col(i),
                "end_line": i.span().end().line, "end_col": i.span().end().column,
                "fn": f, "fn_id": self.cur_fn_id.last(), "cfg_test": cfg}),
            );
        }
        visit::visit_expr_binary(self, i);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        let (f, _, cfg) = self.ctx();
        let init = i.init.as_ref().map(|x| txt(&*x.expr));
        self.lets.push(
            json!({"pat": txt(&i.pat), "init": init, "line": line(i), "col": col(i),
                "end_line": i.span().end().line, "end_col": i.span().end().column, "fn": f, "fn_id": self.cur_fn_id.last(),
                "scope": self.scope, "cfg_test": cfg}),
        );
        visit::visit_local(self, i);
    }
    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        let (f, _, cfg) = self.ctx();
        self.fors.push(
            json!({"pat": txt(&*i.pat), "iter": txt(&*i.expr), "line": line(i), "col": col(i), "fn": f, "fn_id": self.cur_fn_id.last(),
            "cfg_test": cfg}),
        );
        visit::visit_expr_for_loop(self, i);
    }
    fn visit_macro(&mut self, i: &'ast syn::Macro) {
        let (f, _, cfg) = self.ctx();
        self.macros.push(
            json!({"name": txt(&i.path).replace(' ', ""), "line": line(&i.path), "col": col(&i.path), "fn": f, "fn_id": self.cur_fn_id.last(),
            "scope": self.scope, "tokens": i.tokens.to_string(), "cfg_test": cfg}),
        );
        visit::visit_macro(self, i);
    }
}

fn one(path: &str) -> Value {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return json!({"schema": SCHEMA, "file": path, "ok": false, "err": format!("read: {e}")})
        }
    };
    match syn::parse_file(&src) {
        Ok(ast) => {
            let mut v = Facts::default();
            v.visit_file(&ast);
            v.path_roots.sort();
            v.path_roots.dedup();
            v.paths.sort();
            v.paths.dedup();
            v.test_paths.sort();
            v.test_paths.dedup();
            json!({"schema": SCHEMA, "file": path, "ok": true, "file_attrs": attrs_of(&ast.attrs),
                "fns": v.fns, "types": v.types, "mods": v.mods, "uses": v.uses, "consts": v.consts,
                "calls": v.calls, "mcalls": v.mcalls, "macros": v.macros, "index": v.index,
                "assigns": v.assigns, "lets": v.lets, "fors": v.fors, "path_roots": v.path_roots, "paths": v.paths, "test_paths": v.test_paths})
        }
        Err(e) => json!({"schema": SCHEMA, "file": path, "ok": false,
            "err": format!("{}:{} {}", e.span().start().line, e.span().start().column, e)}),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<String> = if args.len() == 1 && args[0] == "-" {
        std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .collect()
    } else {
        args
    };
    for p in paths {
        println!("{}", one(p.trim()));
    }
}
