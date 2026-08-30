' SNDL learn-daemon autostart -- launches the always-on brain learning daemon (hidden) at login.
' Target: learn_daemon.bat in this same directory (full-brain tick + chat->durable-lake + auto-relaunch
' on new commit; see learn_daemon.py).
'
' PROMOTED INTO THE REPO 2026-08-29 (P4c). This file lived in the user's Startup folder, outside the
' tree, in no gate, invisible to every diff -- the same disposition that let learn_daemon.py accumulate
' three latent bugs before its own promotion on 2026-08-04. It is the brain's IGNITION: without it the
' drive never starts at login, which makes it part of the brain by function, whatever directory
' Windows wants a copy in. The Startup folder now holds a one-line shim that runs THIS file, so the
' governed copy is the one that decides anything.
'
' SINGLE-INSTANCE GUARD: if a learn_daemon instance is already running, do nothing -- two daemons
' would contend on the same data/ state (P4: one brain, no replication).
'
' WIDENED 2026-08-29. The guard checked ONLY python.exe. That was adequate while learn_daemon.bat
' retried every 8s: the window in which the supervising loop was alive but python momentarily absent
' was about 8 seconds. The launcher now backs off 8s -> 300s after repeated failures, so that window
' grew to five minutes -- and a login inside it would see no python, start a SECOND launcher loop, and
' end with two supervisors racing to run two brains against one lake. The backoff change made an
' existing race roughly 37x more likely, so the guard is widened in the same breath.
'
' The guard checks BOTH:
'   * python.exe -- a brain is running
'   * cmd.exe    -- a launcher loop is running (brain may be mid-backoff, still supervised)
' Either one means "already supervised; do nothing".
Option Explicit
Dim wmi, procs, p, running, sh, cl, here
running = False
Set wmi = GetObject("winmgmts:\\.\root\cimv2")

' a brain process
Set procs = wmi.ExecQuery("SELECT CommandLine FROM Win32_Process WHERE Name='python.exe'")
For Each p In procs
    If Not IsNull(p.CommandLine) Then
        If InStr(p.CommandLine, "learn_daemon") > 0 Then running = True
    End If
Next

' ...or a launcher loop supervising one (covers the backoff window)
If Not running Then
    Set procs = wmi.ExecQuery("SELECT CommandLine FROM Win32_Process WHERE Name='cmd.exe'")
    For Each p In procs
        If Not IsNull(p.CommandLine) Then
            cl = LCase(p.CommandLine)
            If InStr(cl, "learn_daemon.bat") > 0 Then running = True
        End If
    Next
End If

If Not running Then
    ' Resolve the launcher RELATIVE to this script, so a moved or renamed checkout still starts the
    ' right brain instead of a hardcoded path that silently points at nothing.
    here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
    Set sh = CreateObject("WScript.Shell")
    sh.CurrentDirectory = here
    sh.Run """" & here & "\learn_daemon.bat""", 0, False
End If
