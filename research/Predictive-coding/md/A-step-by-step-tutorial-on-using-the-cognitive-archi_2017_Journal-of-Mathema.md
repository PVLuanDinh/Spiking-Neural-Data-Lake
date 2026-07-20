JournalofMathematicalPsychology76(2017)94–103
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
A step-by-step tutorial on using the cognitive architecture ACT-R in
combination with fMRI data
JelmerP.Borsta,∗ ,JohnR.Andersonb
aUniversityofGroningen,DepartmentofArtificialIntelligence,Nijenborgh9,9747AG,Groningen,TheNetherlands
bCarnegieMellonUniversity,DepartmentofPsychology,5000ForbesAvenue,Pittsburgh,PA15213,USA
h i g h l i g h t s
• WeprovideatutorialonusingthecognitivearchitectureACT-RwithfMRIdata.
• Region-of-interestanalysisandmodel-basedfMRIanalysisarediscussed.
• ROIanalysisisusedtoconstrainandevaluateACT-Rmodels.
• Model-basedfMRIanalysisisusedtofindtheneuralcorrelatesofmodelprocesses.
• Dataandcodetoperformtheanalysesandrecreatethefiguresareprovided.
a r t i c l e i n f o a b s t r a c t
Articlehistory: ThecognitivearchitectureACT-Risatthesametimeapsychologicaltheoryandamodelingframework
Availableonline20June2016 for constructing cognitive models that adhere to the principles of the theory. ACT-R can be used in
combinationwithfMRIdataintwodifferentways:(1)fMRIdatacanbeusedtoevaluateandconstrain
Keywords:
modelsinACT-RbymeansofpredefinedRegion-of-Interest(ROI)analysis,and(2)predictionsfromACT-R
ACT-R
modelscanbeusedtolocateneuralcorrelatesofmodelprocessesandrepresentationsbymeansofmodel-
fMRI
basedfMRIanalysis.Inthispaperweprovideastep-by-steptutorialonbothapproaches.Notethatthis
Cognitivearchitecture
Model-basedanalysis tutorialneitherteachestheACT-Rtheoryinanydetail,norfMRIanalysis,butexplainshowACT-Rcanbe
Tutorial usedincombinationwithfMRIdata.Tothisend,weprovidealldataandcomputercodenecessarytorun
theACT-Rmodel,carryouttheanalyses,andrecreatethefiguresinthepaper.Asanexampledatasetwe
usearelativelysimplealgebratask.Inthefirstsection,wedevelopanACT-Rmodelofthistaskandfitit
tobehavioraldata.Inthesecondsection,weapplyapredefinedROI-analysistoevaluatethemodelusing
fMRIdata.Inthethirdsection,weusemodel-basedfMRIanalysistolocatethefollowingprocessesin
thebrain:retrievalofmathematicalfactsfrommemory,workingmemoryupdates,motorresponses,and
visuallyencodingtheproblems.Afterworkingthroughthistutorial,thereaderwillhavelearnedwhat
canbeachievedwiththetwodifferentanalysismethodsandhowtheyareconducted;theexamplecode
canthenbeadaptedtoanewdataset.
©2016ElsevierInc.Allrightsreserved.
1. Introduction approachtomodelinghascertainlyyieldedsignificantinsights,the
danger is that the mechanisms that are modeled might be task-
Manypsychologicaltheories,andconsequently,manycompu- specific,insteadofrepresentingconceptsthataremoregenerally
tational cognitive models focus on a single task or even on part valid. If the goal is to understand the functioning of the human
ofatask.Forexample,decision-makingmodelsoftenonlysimu- mind–incontrasttounderstandingwhatishappeninginasin-
latethedecision-makingaspectofatask,andinclude‘non-decision gletask–thisapproachmightbecounter-productive,asfamously
time’torepresentallnon-decisionprocesses(e.g.,Forstmannetal., notedbyNewellinhis20-questionspaper(1973).
2008; Turner, van Maanen, & Forstmann, 2015). Although this Incontrasttosingle-taskmodels,cognitivearchitecturestryto
capturegeneralcognitivemechanismsandareaimedatmodeling
completetasksfromperceptiontoresponseexecution(Anderson,
∗ Correspondingauthor. 2007; Newell, 1990). A cognitive architecture is a psychological
E-mailaddresses:j.p.borst@rug.nl(J.P.Borst),ja+@cmu.edu(J.R.Anderson). theory, for instance explaining how our declarative memory
http://dx.doi.org/10.1016/j.jmp.2016.05.005
0022-2496/©2016ElsevierInc.Allrightsreserved.

J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103 95
Fig.1. Setupoftheexperiment.Eachtrialstartedwith4soffixation,followedbythepyramidproblemforamaximumof30soruntilthe‘Submit’buttonwasclicked.
Betweentrialsarepetition-detectiontaskwasdisplayedfor12storeturnthebraintoacommonbaseline.
Source:ReprintedwithpermissionfromAndersonetal.(2011).
system functions (ACT-R; Anderson, Bothell, Lebiere, & Matessa, Matlab (The MathWorks, Inc.), and the fMRI analysis software
1998),howexecutiveprocessesdeterminedual-taskperformance SPM8 (Wellcome Trust Centre for Neuroimaging). We included
(EPIC; Meyer & Kieras, 1997), or how new tasks are learned by all software and toolboxes, except Matlab itself. In Section 2
informationtransfer(PRIMs;Taatgen,2013).Inaddition,cognitive wewillintroducethetaskandtheACT-Rcognitivearchitecture,
architecturesareimplementedascomputersimulations,inwhich anddevelopthecognitivemodel.InSection3wewillapplythe
cognitive models of particular tasks can be instantiated (e.g., predefinedROIanalysis.2 Finally,inSection4wewillusemodel-
recovering from interruptions, Borst, Taatgen, & Van Rijn, 2015; basedanalysistolocatetheneuralcorrelatesofvisuallyencoding
drivingacar,Salvucci,2006;associativerecognition,Schneider& the task, retrieving mathematical facts from memory, updating
Anderson, 2012). Thus, the cognitive architecture itself provides workingmemory,andissuingmotorresponses.Notethatwealso
the general psychological theory – the architecture of the mind provide all intermediate results, in order to allow for starting
– while models run on this architecture to simulate behavior in theanalysisatanypoint,andskippingpartsoftheanalysis.We
particulartasks. indicatedinthetextwhenanalysisstepstakelongerthanafew
Thisapproachhasanumberofadvantages.Mostimportantly, minutes.
it ensures that the cognitive mechanisms used to account for
different tasks are the same, and therefore have to be task- 2. DevelopingtheACT-Rmodel
general. Second, because models cover complete tasks and are
implemented as computer simulations, it is possible to directly 2.1. Thetask
comparetheirresponsetimesandaccuracytohumanparticipants.
Third, because action and perception mechanisms are included Thedatasetthatweuseasanexampleinthistutorialconcernsa
inthemodels,thepredictionsofthesemodels,suchasresponse relativelysimplealgebratask,whichwaspartofalargerstudyused
times,areconstrainedtocorrespondtowhatisphysicallypossible to investigate metacognitive activity in mathematical problem
forhumans(e.g.,Borst,Buwalda,VanRijn,&Taatgen,2013;Meyer solving(Anderson,Betts,Ferris,&Fincham,2011).Inaddition,it
&Kieras,1997). hasbeenusedtovalidatenewregions-of-interestforACT-R(Borst,
Althoughitispossibletoevaluatecognitivearchitecturemod- Nijboeretal.,2015).Wewillfollowtheanalysisstreamandtrial
elsbycomparingtheirresponsetimesanderrorratestohuman selectionfromthislatterpaper.
data, these data are often insufficient to test all the detailed as- In the experiment, participants solved so-called pyramid
sumptionsofthesemodels(problemsofthissorthavebeennoted problems,whichtaketheformbase$height = value,forinstance
withmostkindsofmodeling;e.g.,Pitt&Myung,2002;Roberts& 5$3 = 12(seeAndersonetal.,2011,fordetailsonthestudyand
Pashler, 2000; Shiffrin, Lee, Kim, & Wagenmakers, 2008). To in- therationaleforusingpyramidproblems).Apyramidproblemis
crease the constraints on models in the ACT-R architecture, re- solvedbyaprocessofrepeatedaddition,wherethebaseindicates
searchershaveturnedtofMRIdata.Therearetwowaysinwhich thestartingvalueandtheheight thenumberofadditions,where
ACT-R can be used in combination with fMRI data (Borst & An- eachaddendisonelessthantheprevious:5$3=5+4+3=12.
derson, 2015). First, each ACT-R model predicts fMRI activity in Participantswereaskedtosolvethreetypesofproblems:
severalpredefinedregions-of-interest(ROIs)inthebrain(Ander- • solve-for-value:6$3=x
son,2007;Anderson,Fincham,Qin,&Stocco,2008).Becausethese • solve-for-height:6$x=15
regions are predefined and independent of new datasets – they • solve-for-base:x$3=15.
have remained essentially unchanged since 2003 – such predic-
tionscanbecomparedtohumandatatojudgethequalityofthe Thebaserangedfrom4to9andtheheightfrom2to5,resultingin
modelandguidefurthermodeldevelopment.Second,predictions
valuesbetween4$2=4+3=7and9$5=9+8+7+6+5=35.
fromACT-Rmodelscanbeusedtolocatetheneuralcorrelatesof Foranalysispurposes,wedividedtheproblemsintofourgroups
modelprocessesandrepresentationsthroughmodel-basedfMRI basedonwhethertheyhadasmall(4–6)oralarge(7–9)base,and
analysis(Borst,Taatgen,&VanRijn,2011). whethertheyhadasmall(2–3)oralarge(4–5)height.
Inthistutorial,wewillguidethereaderthroughbothanalysis Twenty subjects participated in the experiment. They each
methods and provide the data and computer code necessary to solved 72 problems in the fMRI scanner, equally divided over
replicatetheanalysesandrecreatethefiguresinthispaper(see conditions. Responses were entered by clicking on an on-screen
http://www.jelmerborst.nl/tutorial).1Toperformtheanalyseswe numeric keypad with the mouse, followed by clicking the
‘submit’button.Aftersubmittingaresponse,subjectsimmediately
usedtheACT-R7.0StandaloneVersion(http://act-r.psy.cmu.edu),
progressedtoafeedbackscreen,followedbyarepetitiondetection
task(Fig.1;detailedmethodsarereportedinAndersonetal.,2011).
1 NotethatthisisneitheratutorialonACT-R,noronfMRIanalysis.Formore
information about ACT-R we refer to act-r.psy.cmu.edu, for more information
on fMRI analyses one can visit for example http://www.fil.ion.ucl.ac.uk/spm/, 2 TheresultsofthisanalysishavebeenpreviouslyreportedinBorst,Nijboer,
http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/,orhttp://www.freesurfer.net. Taatgen,VanRijn,andAnderson(2015).

96 J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103
of3,THENrepresentinthecontrolstatethatapyramidproblemis
beingsolved,andretrieveanadditionfactfromdeclarativememory
with6asthefirstaddendand5asthesecondaddend’.
Eachmoduleadherestoasetofpreciselydescribedproperties
|     |     |     |     |     |     |     | (Anderson,  | 2007), | which     | are instantiated |       | in          | the computational |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | --------- | ---------------- | ----- | ----------- | ----------------- | --- |
|     |     |     |     |     |     |     | simulation. | For    | instance, | it takes         | 85 ms | to visually | process           | an  |
objectonthescreen,andittakes200mstostoreachunkinthe
problemstate.Eachfactindeclarativememoryhasanactivation
level,whichrepresentsthelikelihoodthatitiscurrentlyneeded
|     |     |     |     |     |     |     | (Anderson      | & Schooler, | 1991).        | This   | activation |          | value is dependent |         |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | ------------- | ------ | ---------- | -------- | ------------------ | ------- |
|     |     |     |     |     |     |     | on the recency |             | and frequency | of     | usage      | of the   | chunk, as          | well as |
|     |     |     |     |     |     |     | on the current |             | information   | in the | other      | modules. | The activation     |         |
valuedetermineshoweasyitistoretrieveachunkfrommemory:
thehighertheactivationvalue,themorelikelyitisthatitcanbe
retrieved,andthefasteritwillberetrieved.Forinstance,‘2+2=
|     |     |     |     |     |     |     | 4’ has a | higher | activation | value | and can | be retrieved | faster | than |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------ | ---------- | ----- | ------- | ------------ | ------ | ---- |
‘14×19=266’,ifthelatterisevenrepresentedinmemory.
Fig.2. Responsetimes.Dataontheleft,modelontheright.Thelower,darkerparts
ofthebarsindicatethetimeuntilthefirstclick,thetop,lighterpartsofthebarsthe AllmodulesofACT-Rhavebeenmappedontosmallregionsin
timebetweenthefirstclickandthesubmitbutton.Errorbarsindicatetheaverage
thebrain(Fig.3(b)).Theoriginalmappingwasbasedonareading
standarddeviation.
Source:ReprintedwithpermissionfromBorst,Nijboeretal.(2015). oftheliteratureonregionalfunctions,andadaptedslightlybased
|     |     |     |     |     |     |     | on experience |     | with early | experiments. |     | Since | 2003 the mapping |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------- | ------------ | --- | ----- | ---------------- | --- |
2.2. Behavior has essentially remained unchanged (Anderson, 2007; Anderson
etal.,2008).Asthemoststraightforwardexample,ACT-R’smotor
modulewasmappedontotheregionoftheprecentralgyrusthat
Forthistutorialweusecorrecttrialsonly.Twosubjectswere
representsthehand(Anderson,2005).TheROIsareassumedtobe
excludedforsolvingonly49%and76%oftheproblemscorrectly,
|           |        | >      |             |         |         |            | activewhenthecorrespondingmoduleisactive;thus,themotor |     |     |     |     |     |     |     |
| --------- | ------ | ------ | ----------- | ------- | ------- | ---------- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
| where all | others | scored | 85% (Borst, | Nijboer | et al., | 2015). The |                                                        |     |     |     |     |     |     |     |
.
remaining 18 subjects solved on average 93.4% (SE = 0 9%) of regionshouldbeactivewhenanACT-Rmodelmovesitshand.Note
the problems correctly. Of these problems, we removed outliers thatitisnotclaimedthattheseregionsaretheonlypartsofthe
withresponsetimes(RT)exceeding2standarddeviationsfromthe brainthatareactivewhenamoduleisactive,northattheseregions
meanpersubjectandcondition(4.2%).Thisleaves1160problems areonlyactivewhenamoduleisactive.
Becausetheseregionsarepredefined,andbecauseitisassumed
foranalysisintotal,onaverage64.4problemspersubject.
thattheseregionsarenecessarilyactivewhenamoduleisactive,
TheleftpanelofFig.2showstheresponsetimes.Heighthad
theycanbeusedtoconstrainmodelsbycomparingthepredicted
acleareffectonRTs,withlargeheights(orangebars)resultingin
activityoftheACT-Rmodulestothedataintheseregions(Borst&
longerRTsthansmallheights(bluebars).Asheightdeterminesthe
Anderson,2015;Borst,Nijboeretal.,2015).Forexample,ifACT-R’s
numberoftermsintherepeatedaddition,thisisnotsurprising.
motormoduleisactivebutnoactivityisfoundinthemotorROI,
BasealsohadapositiveeffectonRT:alargebaseresultedinlonger
thisimpliesthatthemodelisincorrect.Inthenextsectionswewill
RTsthanasmallbase.Fig.2showsthatmostofthevariabilityin
demonstratesuchananalysisnotonlyforthemotormodule,but
RTsislocatedinthe‘cognitivephase’(dark),beforethefirstmouse
click,andnotinthe‘responsephase’(light),betweenthefirstclick alsoforthevisualmodule,declarativememory,andtheproblem
andclickingthesubmitbutton.Theseeffectswereconfirmedbya state.Especiallycomparingdifferentconditionlevelsofthemore
|     |     |     |     |     |     |     | central cognitive |     | models | often | provides | important | information |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------ | ----- | -------- | --------- | ----------- | --- |
repeated-measureANOVA(Borst,Nijboeretal.,2015).
aboutthequalityofamodel.
2.3. ACT-R
2.4. Model
| Before | we develop | the | model to | explain | these data, | we will |           |     |                 |     |         |            |             |     |
| ------ | ---------- | --- | -------- | ------- | ----------- | ------- | --------- | --- | --------------- | --- | ------- | ---------- | ----------- | --- |
|        |            |     |          |         |             |         | The model |     | that we present |     | here is | the result | of a series | of  |
firstbrieflyintroduceACT-R.Formoredetail,seeAnderson(2007,
|     |     |     |     |     |     |     | consecutive | efforts | (Anderson, |     | 2007; | Wintermute, | Betts, | Ferris, |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ---------- | --- | ----- | ----------- | ------ | ------- |
book);orAnderson(2005,article);and/orfollowthetutorialunits
|     |     |     |     |     |     |     | Fincham, | & Anderson, | 2012; | the | current | model | was reported |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ----- | --- | ------- | ----- | ------------ | --- |
includedwiththeACT-Rdistribution.3
|       |           |           |           |     |           |          | before in | Borst,  | Nijboer et         | al., 2015, | and | updated | to ACT-R          | 7.0 |
| ----- | --------- | --------- | --------- | --- | --------- | -------- | --------- | ------- | ------------------ | ---------- | --- | ------- | ----------------- | --- |
| ACT-R | simulates | the human | cognitive |     | system as | a set of |           |         |                    |            |     |         |                   |     |
|       |           |           |           |     |           |          | for the   | current | paper). Developing |            | a   | model   | in ACT-R involves |     |
independentmodulesthatarecoordinatedbyacentralproduction
|        |                 |         |      |         |         |         | specifying | the | contents of | procedural | and | declarative | memory— |     |
| ------ | --------------- | ------- | ---- | ------- | ------- | ------- | ---------- | --- | ----------- | ---------- | --- | ----------- | ------- | --- |
| system | (the procedural | module; | Fig. | 3). Two | modules | process |            |     |             |            |     |             |         |     |
theIF-THENproductionrulesandtheinformationthoserulescan
| input from | the | external world | (visual, | aural) | and two | modules |               |         |           |     |           |          |             |     |
| ---------- | --- | -------------- | -------- | ------ | ------- | ------- | ------------- | ------- | --------- | --- | --------- | -------- | ----------- | --- |
|            |     |                |          |        |         |         | use. To solve | pyramid | problems, |     | the model | requires | declarative |     |
send output to the external world (manual, vocal). The other knowledgeaboutadditionproblems,forexample‘6 + 5 = 11’.
modules process information internally: the control state keeps Itthenappliesproductionrulesthatusethisknowledgetosolve
track of the current goal, the problem state holds intermediate pyramid problems. The model is included in the material on
| representations |        | (also referred | to as         | the imaginal | module),    | and |                                     |     |     |     |     |          |            |      |
| --------------- | ------ | -------------- | ------------- | ------------ | ----------- | --- | ----------------------------------- | --- | --- | --- | --- | -------- | ---------- | ---- |
|                 |        |                |               |              |             |     | http://www.jelmerborst.nl/tutorial, |     |     |     | in  | the next | section we | will |
| declarative     | memory | stores         | facts (called | chunks).     | Information | is  |                                     |     |     |     |     |          |            |      |
explainhowtorunthemodelandgeneratepredictions.
processedbyproductionrulesinproceduralmemory,whichareIF-
Themodelstartsbyreadingthepyramidproblemonthescreen
THENrules.Forexample,aproductionrulemightsay‘IFthereisa
usingthevisualmodule.Itencodesthebase,height,andvaluein
pyramidprobleminthevisualmodulewithabaseof6andaheight
turn,andfollowsitsowngazewiththemousepointer(cf.Chen,
Anderson,&Sohn,2001).Basedonwhichofthetermsreadx,it
goesdownoneofthreesolutionpaths.Ifthevalueisunknown,it
followsaprocessofiterativeaddition.Itstartsbyaddingthebase
3 ThelatestversionofACT-Rcanbedownloadedfromhttp://act-r.psy.cmu.edu/;
tothe‘base−1’.Tosolvesuchanaddition,themodelretrievesthe
theversionthatweusedisincludedinthematerialonhttp://www.jelmerborst.nl/
solutionfromdeclarativememory,whichtakesacertainamountof
tutorial.

J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103 97
Fig.3. TheACT-Rcognitivearchitecture(a)anditsmappingtobrainregions(b).Thecolorsofthemodulescorrespondtothecoloredsquaresinthebrain.(Forinterpretation
ofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
Source:ReprintedwithpermissionfromBorstandAnderson(2015).
time(estimatedfromthebehavioraldata).Itrepeatsthisprocess betweenstudies,theycanbeusedtoevaluatenewmodelsbycom-
untilithasaddedasmanytermsastheheight indicated.Itthen paring model predictions to human data (Anderson et al., 2008;
continues to the response phase. If the height is unknown, the Borst,Nijboeretal.,2015).Inthissectionwewillprovideastep-by-
model follows the same process, but terminates when the total steptutorialofthismethod.First,wewillusethemodeltogener-
sumisequaltothevalue,andcontinuestotheresponsephaseto atefMRIblood-oxygen-leveldependent(BOLD)signalpredictions
reportthenumberofadditionsithadtoperform.Themostdifficult
forfourmodules:visual,manual,declarativememory,andprob-
caseiswhenthebaseisunknown.Subjectsreportedusingaguess-
lemstate.Second,wewillselectthecorrespondingfMRIdataand
and-checkprocedureinthiscase(Andersonetal.,2011),whichis
evaluatethemodel.
simulatedbythemodel.Themodelfirstguessesabase,andthen
followstheunknown-valueproceduredescribedabovetocalculate
thecorrespondingvalue.Ifthisisequaltothevalueonthescreenit 3.1. Generatingmodelpredictions
reportstheguessedbase,ifthecalculatedvalueissmalleritreports
the‘guessedbase+1’,ifitislargeritreportsthe‘guessedbase−1’. ThefMRIBOLDsignal–whichwesimulatewiththemodel–
Afterthemodeldeterminedasolutionbyfollowingoneofthese isaveryslowsignal,asfMRIscannersmeasuretheoxygenation
procedures,itlooksatthebuttonwiththefirstdigitofitsanswer, leveloftheblood.Itpeaksaround4–6safterneuronalactivity,and
movesthemousecursortothisbutton,clicksit,andreadswhatit canbedescribedbyaso-calledhemodynamicresponsefunction
entered.Itthenmovesontothenextdigit,andcontinuesinthis (HRF;Fig.4(a);e.g.,Friston,Ashburner,Kiebel,Nichols,&Penny,
manneruntilthefullanswerisentered.Itfinishesbyclickingthe
2007). In contrast, the model’s activity occurs immediately and
‘Submit’button,andreadingthefeedbacktodeterminewhetherit
isthereforemorecomparabletoneuralactivity.Tosimulatethe
wascorrect.
BOLDresponse,onecanconvolvethemodel’sactivityonamodule-
The right panel of Fig. 2 shows the response times of the
by-module basis with an HRF (Fig. 4(b); Anderson et al., 2008;
model. These times are a combination of all the processes the
Borst&Anderson,2015).Inessence,thismathematicalprocedure
model has to go through: visual perception, memory retrievals,
assumesahemodynamicresponseforeachactivityofamodule,
production rules, and manually moving the mouse and entering
the response. The duration of the perception and response andtheseresponsesaresummedtoproducethefinalsignal.The
processes are relatively constant over the different trial types, processconsistsoftwosteps.
but the length of the addition process varies considerably with Step1.Collectingmoduleactivity.First,werunthemodelon
thedifficultyoftheproblems.First,largerheightsresultinmore thesamepyramidproblemsthatthesubjectsperformed.Thefile
additionsteps, andthusinlonger totalRTs. Second,largerbases containing the model code, ‘pyramid-model_jmp.lisp’, is located
result in harder additions that include more carries, which add in the ‘model’ directory. This file contains Lisp code to run the
additional computation steps to the model (Borst, Nijboer et al., experiment4andtheACT-Rmodelcode(startingatline548).You
2015).Overall,themodelreflectedthedataclosely—thegreatest canopenthisfileinatexteditortotakeafirstlookatthemodel.
discrepancy was the underestimation of the variability of the TodevelopmodelsinACT-R,itisgenerallyrecommendedtouse
cognitive phaseby themodel (i.e.,the standarddeviation of the
afull-fledgedLispenvironmentthatprovidesaGUIanddebugging
model’s response times is smaller than of the human subjects). tools.5 However, the ACT-R Standalone version includes a Lisp
In the next section we will investigate whether the model can
distributionthatcanalsobeusedtorunmodels.Thisistheversion
alsoaccountforthefMRIdatathatwascollectedduringthistask
includedwiththistutorial.TostartACT-R,navigatetothe‘ACT-
(Andersonetal.,2011),whichwouldprovideadditionalevidence
RStandaloneMac’or‘ACT-RStandaloneWindows’directoryand
that the model is a good simulation of the cognitive processes
double click on the file ‘Run ACT-R.command’ on a Mac or on
underlyingthistask.
3. PredefinedROIanalysis
4 ACT-RiswritteninLisp,andforthatreasonexperimentalcodethatACT-R
Asexplainedabove,allmodulesoftheACT-Rarchitecturehave
interfaceswithisalsotypicallywritteninLisp.ACT-Rcontainsasetoffunctions
beenmappedontosmallregions-of-interestofthebrain,topro- thatcanbeusedtoimplementsimpleexperiments.
videadditionalconstraintsformodelsdevelopedinthearchitec- 5 ForexampleAllegroCommonLisp(http://franz.com)orLispworks (http://
ture. Because these ROIs are predefined and remain unchanged www.lispworks.com)bothhavefreeversionsavailable.

98 J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103
Fig.4. (a)Hemodynamicresponsefunction(HRF)asusedbySPM.(b)Declarativememory(top)andmanualmodule(bottom)demandfunctionsandpredictedBOLD
responseforasingletrialoftheexperiment.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
‘Start Environment.exe’ on Windows.6 Click on the ‘Load Model’ these demand functions with the HRF commonly used in SPM
button in the environment, and select the model file (‘pyramid- (Fristonetal.,2007).ThecodetodothisisincludedintheMatlab
script‘ROIanalysis/convolution.m’.Youcanfollowtheinstructions
model_jmp.lisp’).Thisloadsboththeexperimentalcodeaswellas
theACT-RmodelitselfintoLisp.Tousethemodelinafull-fledged in this file to recreate Fig. 4. Note that you have to change the
Lispenvironment,firstloadACT-R,andthenthemodel.7
workingdirectorytothelocationofthe‘ROIanalysis’folderonyour
Todisplaytheexperimentalinterfaceandhavethemodelsolve computeratthebeginningofthefile.First,theHRFfromSPMis
fourproblemsinreal-time,intheListenerwindowtype loadedandplotted(Fig.4(a)).Next,themoduleactivityforonetrial
(run-4-trials) isreadin,representedasa0–1demandfunction,convolvedwith
and hit the ‘return/enter’ key. When the model has finished theHRF,anddisplayed(Fig.4(b)).
running,atthebottomoftheListenerwindowyouwillsee‘NIL’. Fig.4(b)showstheresultsofconvolvingasingletrialofthelarge
To have the model perform all trials of all subjects virtually (no base–largeheightcondition.Severaldeclarativememoryretrievals
window is displayed and the simulation runs as fast as your occur in quick succession and lead to one large bump in the
computer allows), and save the results to disk, navigate to the predicted BOLD profile. The manual module shows a different
Listenerwindowandtype pattern, with mouse movements at the start of the trial (while
(run-real-problems). readingthepyramidproblem)andattheendofthetrialtoenter
|               |          |                  |              |        |                  | the response. | This results            | in a very | different | BOLD prediction |     |
| ------------- | -------- | ---------------- | ------------ | ------ | ---------------- | ------------- | ----------------------- | --------- | --------- | --------------- | --- |
| The model     | will now | run through      | all subjects |        | and trials. This |               |                         |           |           |                 |     |
|               |          |                  |              |        |                  | with two      | clearly distinguishable | bumps.    | Generally | speaking,       | the |
| takes between | 2 and    | 6 min, depending |              | on the | speed of your    |               |                         |           |           |                 |     |
computer. The resulting behavior is saved to ‘model_#.txt’ in convolutionwiththeHRFcombinesandsmoothsoutseveralshort
‘ACT-R Standalone/model_data/’, where ‘#’ indicates the subject moduleactivities.ThisisaprobleminherenttofMRI,andprevents
number. The corresponding module activity that we need for usfromgettingadirectmeasurementofmoduleactivity.However,
generatingfMRIpredictionsissavedto‘model_act_#_#_#.txt’in by using the right experimental designs, one can still make a
‘ACT-RStandalone/model_data_mri/’,wherethe#-marksindicate distinctionbetweenseveralpossiblemodels.Next,wewillperform
subject,block,andtrial,respectively. theconvolutionforalldifferentsubjects,trials,andconditions,and
FormoreinformationontheACT-RtheoryandrunningACT-R seehowtheBOLDresponsepredictionsreacttotheexperimental
| models,seethetutorialunitsincludedwiththeACT-Rdistribution, |     |     |     |     |     | manipulations. |     |     |     |     |     |
| ----------------------------------------------------------- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- |
orvisittheannualsummerschoolatCarnegieMellonUniversity Fig. 5 shows the average of the predictions for four modules,
(Pittsburgh,USA)orspringschoolattheUniversityofGroningen split out by experimental condition. This analysis is also part of
|     |     |     |     |     |     | the Matlab | ‘convolution.m’ | script, starting | at line | 127. All | of the |
| --- | --- | --- | --- | --- | --- | ---------- | --------------- | ---------------- | ------- | -------- | ------ |
(Groningen,theNetherlands).
|         |            |        |               |     |                | model activity | (for all | subjects and trials) | is read | in, convolved |     |
| ------- | ---------- | ------ | ------------- | --- | -------------- | -------------- | -------- | -------------------- | ------- | ------------- | --- |
| Step 2. | Convolving | module | activity with | the | HRF. The files |                |          |                      |         |               |     |
withtheHRF,andaggregated.Thistakesbetween1and10min,
thatweregeneratedinStep1containthetimesinatrialthatACT-
dependingonthespeedofyourcomputer.Fig.5showstheresults.
R’smoduleswereactive.Eachlineinafileliststhemodulename,
|     |     |     |     |     |     | Starting at | the top left, | the predictions | for declarative | memory |     |
| --- | --- | --- | --- | --- | --- | ----------- | ------------- | --------------- | --------------- | ------ | --- |
thestarttime,andtheendtime.8Tomakethesedirectmeasures
of activity comparable to the slow BOLD response, we convolve show a clear increase with condition, with the largest effect for
themwithanHRF(Fig.4).First,werepresentmoduleactivityas height.ThiseffectmirrorstheRTeffect,andisexplainedsimilarly:
a 0–1 demand function (blue line, Fig. 4(b)). Next, we convolve themoredifficultpyramidproblemsrequireadditionaldeclarative
retrievals.ACT-R’sproblemstatemoduleshowsasimilarpattern.
Itisusedtostoreintermediatesolutionstothepyramidproblems,
|     |     |     |     |     |     | and has to | be updated | more often for | more difficult | problems. |     |
| --- | --- | --- | --- | --- | --- | ---------- | ---------- | -------------- | -------------- | --------- | --- |
6
IncasetheACT-Rversionincludedisnotcompatiblewithyourmachine,you Themanualandvisualmodulesshowoppositepatterns:theeasier
candownloadthelatestversionfromhttp://act-r.psy.cmu.edu/. the problems, the higher and narrower the BOLD predictions.
7
Whenusingafull-fledgedLispenvironment,starttheACT-Rinterfacebycalling The explanation is straightforward: according to the model, the
(run-environment)beforerunningthemodel.WithouttheACT-Rinterface,the
|     |     |     |     |     |     | same number | of mouse | and eye movements |     | has to be | made |
| --- | --- | --- | --- | --- | --- | ----------- | -------- | ----------------- | --- | --------- | ---- |
modelmightnotrunbecauseofissueswithbuttonplacementsinthegraphical
|     |     |     |     |     |     | in the different | conditions, | but they are | spaced | further apart | in  |
| --- | --- | --- | --- | --- | --- | ---------------- | ----------- | ------------ | ------ | ------------- | --- |
windows.
8 the more difficult conditions because the mathematical solving
Notethatthesetimeswillbeslightlydifferenteachtimethemodelisrun,as
|     |     |     |     |     |     | phase is longer. | As a | result, the conditions | with | large | heights |
| --- | --- | --- | --- | --- | --- | ---------------- | ---- | ---------------------- | ---- | ----- | ------- |
declarativememoryprocesseshaveanassociatednoisecomponent.

J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103 99
Table1
ACT-Rmodulesandassociatedbrainregions.
ACT-Rmodule Brainregion(lefthemisphere) Talairach–Tournouxcoordinates MNIcoordinates
Declarativememory Inferiorfrontalsulcus(BA45/46) −42,23,24 −43,24,25
Problemstate Intraparietalsulcus(BA7/39/40) −24,−63,40 −24,−67,44
Manual Precentralgyrus(BA3) −42,−20,50 −42,−23,54
Visual Fusiformgyrus(BA37) −41,−61,−9 −43,−60,−16
MNI=MontrealNeurologicalInstitute.
Fig.5. ModelpredictionsforfourACT-Rmodules. Fig.6. DatainfourROIs,correspondingtothemodelpredictionsinFig.5.
showtwobumps:onebeforethesolvingphaseandoneafter(cf. experimental design, one can also calculate %-change compared
Fig.4(b),bottom).However,theareaunderthecurveissimilarin toonlythefirstscanortolongerfixationperiodsatthestartof
allconditions,reflectingtotalactivity(itisalittlelargerinthemore ablock.Calculating%-changeonatrial-by-trialbasisensuresthat
difficultconditions,becausetheanswersrequiremorekeypresses slowsignalchangesareremoved(instandardfMRIanalysisahigh-
intheseconditions).Inthenextsectionwewilltestifthedatais passfilteristypicallyappliedforthispurpose).Finally,thedataare
similartothemodel’spredictions. averagedwithineachROIandcondition.Theresultsareshownin
Fig.6.
3.2. SelectingfMRIdata
3.3. Discussion
Toselectthedata,weusepreprocessedfMRIscans(realigned,
normalized, smoothed; Borst, Nijboer et al., 2015) and a defi- ComparingthedatainFig.6tothemodelpredictionsinFig.5
nition of the ROIs. The selection is done using SPM8 in com- shows that the model gives a good qualitative account of the
bination with the MarsBaR toolbox (Brett, Anton, Valabregue, data: the order of the conditions was correctly predicted for
& Poline, 2002). The preprocessed fMRI scans, the ROI defini- all four modules. In the declarative memory module, however,
tions, SPM8 and MarsBaR are all included in the material on thereishardlyadifferencebetweensmallandlargebasesforthe
http://www.jelmerborst.nl/tutorial.ThepropertiesoftheROIsare largeheights(orangelines)wherethemodelpredictedasizeable
listed in Table 1, the ROIs are shown in Fig. 3(b), and MarsBaR difference.Theproblemstatemodulehasaverygoodfitonfirst
compatible ROI files are included in ‘ROIanalysis/rois/’. Each ROI sight,butitturnsoutthatinthedatatheactivityismoreprolonged
is15 × 15 × 12mm.Thepreprocessedscansarelocatedinthe (i.e. the curves return to baseline only at 20 s, in the model
‘data’directory,includingadesignfile‘pyramids.df’thatdescribes predictions at 15 s). This might indicate that working memory
whichscansbelongtowhichsubject,trial,andcondition.TheMat- also plays a role in processing the feedback – which the model
labscript‘ROIanalysis/roi_data_selection.m’containsthecodeand ignored–,notanunreasonableassumption.Inaddition,itcould
instructionstoselectthedataandrecreateFig.6. reflect the greater variability of the RT’s in the data. Finally, the
First,allscansofasubjectarereadinandscaledtoanaverage manualandvisualpredictionsaregenerallyverygood,exceptthat
value of 100 over all scans and voxels, as is common in fMRI thedifferencebetweenthesmall$largeandlarge$largeproblems
analysis (e.g., Friston et al., 2007). Second, the data in each ROI islessclearinthedata,whichmightbeduetothesmallervariation
is collected, and a mean value per ROI is calculated. These two inthemodel’sRTs(Fig.3).
stepstakebetween5and36min.Third,thepercentBOLDchange Quantitatively,largedifferencescanbeobservedbetweenthe
with respect to the average of the first two scans in a trial model and data. The ACT-R model predictions do not take into
(duringthefixationscreen,Fig.1)iscalculated.Dependingonyour accountdifferencesintheHRFbetweenregions,andaregenerally

100 J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103
badpredictorsofamplitude.Asasolution,onecanfitaseparate Intheremainderofthissectionwewilldemonstrateamodel-
scaleparametertoeachmodule.Forinstance,Anderson,Qin,Jung, basedanalysisofthepyramiddataset.Wewillattempttolocate
|            |        |           |         |           |     |          |         | declarative | memory  | retrievals, |             | updates | to the | problem | state,  |
| ---------- | ------ | --------- | ------- | --------- | --- | -------- | ------- | ----------- | ------- | ----------- | ----------- | ------- | ------ | ------- | ------- |
| and Carter | (2007) | estimated | a scale | parameter |     | of 2.38% | for the |             |         |             |             |         |        |         |         |
|            |        |           |         |           |     |          |         | manual      | actions | and, visual | perception. |         | As the | first   | step we |
visualmodule,3.16%forthemanualmodule,1.79%fortheproblem
state, and 1.15% for declarative memory. Although multiplying generatemodulepredictionsandrescalethemwithrespecttothe
thecurrentpredictionsinFig.5withthesevalueswouldleadto behavioraldata.Next,wefitaGLManddiscusstheresults.Note
a better fit, it would by no means remove the whole difference that we included the intermediate results of all analysis steps,
between Figs. 5 and 6. So far, we have not been able to find makingitpossibletoskipstepsorsubjectsintheanalysis,assome
stable scale parameters (see e.g., Stocco & Anderson, 2008, for oftheanalysesbelowtakeasignificanttime.
| very different | estimates), |     | and have | therefore |     | concentrated | on  |     |     |     |     |     |     |     |     |
| -------------- | ----------- | --- | -------- | --------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
qualitativepredictions. 4.1. Generatingmodelpredictionsandrescaling
| Based        | on the  | current | fit, we | argue     | that | the model         | gives |       |             |     |               |     |           |     |          |
| ------------ | ------- | ------- | ------- | --------- | ---- | ----------------- | ----- | ----- | ----------- | --- | ------------- | --- | --------- | --- | -------- |
|              |         |         |         |           |      |                   |       | For a | model-based |     | fMRI analysis | on  | the basis | of  | an ACT-R |
| a reasonable | account | of      | what is | happening |      | when participants |       |       |             |     |               |     |           |     |          |
solve pyramid problems. Several things could be improved, as model, we use module activity patterns as regressors on a trial-
indicated above, but the basic cognitive processing is accounted by-trialbasis.Thus,themodelisrunontheexactsametrialsasthe
for.However,inothercaseswehavehadtoabandonafirstmodel, participants.AnalogoustotheROIanalysis,theresultingactivity
andproposedifferentprocessingstreamstoaccountforthedata— patternsarefirstconvolvedwithanHRF(Fig.4),andthenregressed
processingstreamsthatwereinspiredbytheneuraldata(Nijboer, against the fMRI data. However, before the GLM can be fit, one
Borst,VanRijn,&Taatgen,2016). additionalstephastobeperformed:themodelpredictionshave
tobematchedtotheparticipants’behavioronatrial-by-trialbasis.
Insummary,theROIanalysiscanbeusedtoevaluatemodels
Thatis,ifaparticipanttook6stogivearesponseonaparticular
| because | we use | predefined | ROIs, | and do | not | change | these ROIs |     |     |     |     |     |     |     |     |
| ------- | ------ | ---------- | ----- | ------ | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
trialwhilethemodelpredicted10s,themodelpredictionshave
onastudy-by-studybasis.AlthoughthecurrentROIsarebasedon
|     |     |     |     |     |     |     |     | to be scaled | back | to the | 6-s duration. | Otherwise, |     | we  | might be |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ------ | ------------- | ---------- | --- | --- | -------- |
theneuroscientificliterature,wehaverecentlyperformedalarge
|     |     |     |     |     |     |     |     | comparing | predicted | module | activity | to, | for instance, |     | a fixation |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | ------ | -------- | --- | ------------- | --- | ---------- |
data-drivenmeta-analysisinwhichwehaveshownthattheseROIs
screen(Borstetal.,2011;Gläscher&O’Doherty,2010).Toavoid
arealsoclosetothebestpredictorsoftheACT-Rmodules(Borst&
|     |     |     |     |     |     |     |     | such problems, |     | all module | activities | are | transformed |     | linearly – |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ---------- | ---------- | --- | ----------- | --- | ---------- |
Anderson,2013;Borst,Nijboeretal.,2015).Inthenextsectionwe
|     |     |     |     |     |     |     |     | either increased |     | or decreased | in  | duration | – to | create | a perfect |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | --- | -------- | ---- | ------ | --------- |
discussasecondtechnique,model-basedfMRIanalysis,whichcan
matchtothebehavioraldata.ThisprocessisillustratedinFig.7.
beusedtolocatemodelconstructsinthebrain.
TheresultingactivityisusedtofittheGLM.
Notethatsucharescalingprocesscouldalsohavebeenapplied
4. Model-basedfMRIanalysis
|         |          |          |     |          |         |         |      | in the ROI | analysis  | in  | the previous   | section.  | This       | would       | have |
| ------- | -------- | -------- | --- | -------- | ------- | ------- | ---- | ---------- | --------- | --- | -------------- | --------- | ---------- | ----------- | ---- |
|         |          |          |     |          |         |         |      | provided   | a perfect | RT  | fit,9 ensuring | that      | behavioral | differences |      |
| The ROI | analysis | requires | an  | existing | mapping | between | ACT- |            |           |     |                |           |            |             |      |
|         |          |          |     |          |         |         |      | between    | the model | and | data did       | not cause | the        | differences | in   |
R modules and brain regions, and has the goal to constrain the fMRI fit (Figs. 5 and 6). However, this has not been done
models. However, at other times we want to discover new traditionally,andcouldnotbedoneinthecaseoftestingtruea
connectionsbetweenmodelconstructsandbrainareas,orevaluate priorimodelpredictions(Borst,Taatgen,Stocco,&VanRijn,2010).
existingmappings.Tothisendonecanapplymodel-basedfMRI WedidnotapplythisrescalingprocesstotheROIanalysisabove,
analysis(Borstetal.,2011;Gläscher&O’Doherty,2010;O’Doherty, aswewantedtoillustratetheconnectionbetweenRTpredictions
Hampton,&Kim,2007).InastandardfMRIanalysis,onetypically andfMRIpredictions.
uses the condition-structure of the experiment as the basis for Step1.Collectingmoduleactivity.Thisstepisidenticaltothe
regressors in a General Linear Model (GLM; e.g., Friston et al., onefortheROIanalysisinSection3.1.However,whilefortheROI
2007). For each voxel (a pixel in three-dimensional space), an analysis a model is not necessarily run on the exact same trial
independent linear model is fit using these regressors (after sequenceastheparticipantsreceived,thisisrequiredforamodel-
convolvingtheregressorswithanHRF,cf.Fig.4),andconditions basedanalysis,becausematchingpredictionsareneededforeach
arecomparedtodetermineinwhichvoxelstheBOLDleveldiffers trial.Asbefore,openACT-R,loadthemodel,andtype
significantlybetweenconditions.Thedrawbackofthisapproach (run-real-problems)
is that a cognitive function is assumed to be ‘on’ or ‘off’ in each to generate all model predictions. See Section 3.1 for detailed
| condition. | In addition, | especially |     | with tightly |     | coupled | processes | instructions. |     |     |     |     |     |     |     |
| ---------- | ------------ | ---------- | --- | ------------ | --- | ------- | --------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
such as declarative memory retrievals and working memory Step 2. Rescaling. To perform the linear transformation and
updating,itisverydifficulttohaveoneprocesswithouttheother, recreate Fig. 7, open ‘ModelBasedAnalysis/model_predictions.m’.
Atthestartofthisfile,Fig.7isrecreatedtoillustratetherescaling
makingithardtoisolatesuchfunctions.
|             |     |               |            |     |      |         |          | process. | First, the | behavior | of the | model | and the | participant | is  |
| ----------- | --- | ------------- | ---------- | --- | ---- | ------- | -------- | -------- | ---------- | -------- | ------ | ----- | ------- | ----------- | --- |
| Model-based |     | fMRI analysis | alleviates |     | this | problem | by using |          |            |          |        |       |         |             |     |
readin,aswellasthemoduleactivityforthisparticulartrial.The
| predictions | of a | computational | model | as  | regressors | in  | the GLM. |        |            |          |          |               |     |     |           |
| ----------- | ---- | ------------- | ----- | --- | ---------- | --- | -------- | ------ | ---------- | -------- | -------- | ------------- | --- | --- | --------- |
|             |      |               |       |     |            |     |          | module | activities | are then | rescaled | by increasing |     | the | durations |
Thesepredictionscantaketheformofparametervaluespertrial
|     |     |     |     |     |     |     |     | and changing | the | onsets, | on the | basis of | the RT-difference. |     | This |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------- | ------ | -------- | ------------------ | --- | ---- |
(e.g.,Daw,Gershman,Seymour,Dayan,&Dolan,2011;Gläscher&
O’Doherty,2010;O’Dohertyetal.,2007;vanMaanenetal.,2011) is done separately for the first key press and the time between
orofactivityofhigher-levelprocesses(i.e.moduleactivity;Borst thefirstkeypressandpressingthesubmitbutton.Notethatthe
&Anderson,2013;Borstetal.,2011).AsinstandardfMRIanalysis, resultingmodelfitstheRTdataperfectly,aswasthegoalofthis
procedure.
| a linear | model | is fit for | each voxel. | Voxels | in  | which | the BOLD |       |            |      |                  |     |           |     |            |
| -------- | ----- | ---------- | ----------- | ------ | --- | ----- | -------- | ----- | ---------- | ---- | ---------------- | --- | --------- | --- | ---------- |
|          |       |            |             |        |     |       |          | After | recreating | Fig. | 7, the rescaling |     | procedure | is  | applied to |
activationcorrelatessignificantlywiththemodelpredictionsare
|              |     |             |     |               |     |          |         | all trials | of all    | participants,    | and | the results | are                   | saved | to SPM- |
| ------------ | --- | ----------- | --- | ------------- | --- | -------- | ------- | ---------- | --------- | ---------------- | --- | ----------- | --------------------- | ----- | ------- |
| then assumed | to  | be involved | in  | the simulated |     | process. | Using a |            |           |                  |     |             |                       |       |         |
|              |     |             |     |               |     |          |         | compatible | ‘multiple | condition’-files |     | in          | the ‘ModelBasedAnaly- |       |         |
modelhastheadvantagethatadifferentpredictorcanbeusedfor
|     |     |     |     |     |     |     |     | sis/design_SPM’ |     | directory. | The | code for | this is | also included | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ---------- | --- | -------- | ------- | ------------- | --- |
eachtrial,andthatmultipleprocessescanbeactivesimultaneously
| and to different |      | degrees  | within | each trial, | making | model-based |        |     |     |     |     |     |     |     |     |
| ---------------- | ---- | -------- | ------ | ----------- | ------ | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| fMRI analysis    | more | powerful | than   | standard    | fMRI   | analysis    | (Borst |     |     |     |     |     |     |     |     |
| etal.,2011).     |      |          |        |             |        |             |        | 9   |     |     |     |     |     |     |     |
Naturally,theresultingRTscouldnotbeusedtoconstrainthemodel.

J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103 101
Fig.7. Rescalingmoduleactivityformodel-basedfMRIanalysis.Inthefirstrow,theapriorimodulepredictionsforthemanualmodule(orange)anddeclarativememory
(blue)areshown(cf.Fig.4(b)).Themodel’skey-pressesareindicatedbythedashedlines.Inthesecondrow,thesemodulepredictionshavebeenlinearlytransformedto
matchtheactualbehavioronthistrial(indicatedbythesolidlines).Notethatthischangesboththeonsetsaswellasthedurationsofthemodule’sactivities.Inthisexample
themoduleactivitieswereincreasedinduration,butincaseswherethemodel’spredictionsweretooslow,theseactivitiesweredecreasedinduration.(Forinterpretation
ofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
‘model_predictions.m’,startingatline187.Theresultingfilescon- of a recent meta-analysis of five model-based analyses (Borst
tain four regressors that are used for the GLM: one for declara- & Anderson, 2013). Especially the results of the problem state
tivememory,onefortheproblemstate,oneforthemanualmod- moduleandthemanualmodulearedifferentthanexpected(note
ule,andoneforthevisualmodule.Theseregressorsaredescribed thatthet-valueswerealsomuchlowerforthosemodules).This
inthreevariables:names,onsets,anddurations.Theconvolution emphasizes the importance of proper experimental design: The
withtheHRFwillbedoneautomaticallyinSPMwhentheGLMis currentstudywasnotdesignedforamodel-basedanalysis,andthe
fit. predicted activity was therefore very highly correlated between
modules (Fig. 5), making it problematic to isolate a particular
4.2. FittingtheGLM module. If one wants to apply model-based analysis to locate a
certainmodelprocess,itisimportanttovarytheuseofthisprocess
TofittheGLMweusedSPM8,whichisaMatlabsoftwarepack- whilekeepingotherprocessesconstant.Anotheroptionistouse
ageforanalyzingfMRIdata.TheMatlabscript‘ModelBasedAnaly- ameta-analysisoverdifferentexperiments,becauseitavoidsthe
sis/model_based_analysis.m’usesSPM’sbatchsystemtofitaGLM high inter-module correlations that often occur in a particular
foreachsubjectseparately,usingtherescaledmodelpredictions experiment.
fromthepreviousstep,aswellasmovementregressorsfromre-
aligning the scans. This takes between 20 min and 3 h 40 min 5. Finalremarks
(1–11minpersubject).Foreachsubject,SPMwillaskifyouwant
to overwrite the subject, because we included the results of the In this paper we provided a tutorial of two different ways in
analysisaswell(werecommendchoosing‘continue’forthefirst whichACT-RmodelscanbeusedwithfMRIdata.Thepredefined
subjecttoobservetheprocess,afterwardsyoucanchoose‘stop’to ROI analysis can be used to constrain ACT-R models and guide
skiptherestofthesubjectsandmoveontothesecond-levelanaly- modeldevelopment,whilemodel-basedfMRIanalysiscanbeused
sis).SPMthenperformsasecond-levelanalysis(startingatline104 tolocateneuralcorrelatesofmodelprocesses.Notethatinmost
of‘/model_based_analysis.m’;thistakesbetween30sand6min) situationsonewouldnotperformbothanalysesonasingledataset
anddisplaystheresultsofFig.8.Notethattheseanalysescanbe (atleastnotforthesamemodules):afteradjustingamodulebased
performedwithanyfMRIanalysissoftwareandarenotrestricted on an ROI analysis it would be obvious to find the ROI used for
toSPM. theadjustmentsinamodel-basedanalysis.Finally,theseanalyses
arenotlimitedtoACT-Rmodels.Whilemodel-basedanalysiswas
4.3. Discussion originally used for other types of models (e.g., Daw et al., 2011;
O’Dohertyetal.,2007;vanMaanenetal.,2011),theROIanalysis
Fig. 8 shows the results for all four modules. At the top could also be applied to other modeling paradigms. In addition,
significantregionsfordeclarativememoryareshown.Inessence, whileweusedMatlabandSPMassoftwareinthispaper,thesame
the fronto-parietal network was found, which is a sensible analysescouldbeperformedwithdifferentsoftware.
result as this network is known to be important for controlled
episodic memory retrievals (Borst & Anderson, 2013; Cabeza
Acknowledgments
et al., 2003; Cole & Schneider, 2007; Dosenbach et al., 2007;
Spreng,Stevens,Chamberlain,Gilmore,&Schacter,2010).ACT-R’s
WethankLaelSchooler,IoannaKatidioti,andTrudyBuwalda
predefinedregionfordeclarativememoryispartofthisnetwork
fortestingthetutorialcode.ThisresearchwassupportedbyNWO
(around the inferior frontal sulcus, slice +28, see also Fig. 3(b)).
Venigrant451-15-040awardedtoJelmerBorst,andbyERC-StG
However,theparietalpartofthenetwork(slices+40and+52)
grant283597awardedtoNielsTaatgen.
isnormallyattributedtoACT-R’sproblemstatemodule.Thiswas
notconfirmedbythecurrentmodel-basedanalysis:problemstate
activitywaslimitedtothesupplementarymotorarea,leftinferior References
frontalgyrus,andleftandrightprecuneus.Asexpected,theright-
Anderson,J.R.(2005).Humansymbolmanipulationwithinanintegratedcognitive
handed mouse movements of the model were mapped onto the
architecture.CognitiveScience,29,313–341.
leftmotorcortex,butalsotothemedialfrontalgyrusandtoalarge Anderson,J.R.(2007).Howcanthehumanmindoccurinthephysicaluniverse?.New
areaintheoccipitalcortex.Finally,thevisualmodulemappedmost York:OxfordUniversityPress.
stronglyontoextendedactivationintheoccipitalcortex. Anderson, J. R., Betts, S., Ferris, J. L., & Fincham, J. M. (2011). Cognitive and
metacognitive activity in mathematical problem solving: prefrontal and
Although these results are generally sensible, they are not parietalpatterns.Cognitive,Affective,&BehavioralNeuroscience,11(1),52–67.
fully in line with ACT-R’s predefined mapping and with results http://dx.doi.org/10.3758/s13415-010-0011-0.

102 J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103
Fig.8. Resultsofthemodel-basedanalysis,thresholdedatp<0.01andatleast200contiguousvoxels.
Anderson,J.R.,Bothell,D.,Lebiere,C.,&Matessa,M.(1998).Anintegratedtheory Daw,N.D.,Gershman,S.J.,Seymour,B.,Dayan,P.,&Dolan,R.J.(2011).Model-based
oflistmemory.JournalofMemoryandLanguage,38,341–380. influencesonhumans’choicesandstriatalpredictionerrors.Neuron,69(6),
Anderson,J.R.,Fincham,J.M.,Qin,Y.,&Stocco,A.(2008).Acentralcircuitofthe 1204–1215.http://dx.doi.org/10.1016/j.neuron.2011.02.027.
mind.TrendsinCognitiveSciences,12(4),136–143.
Dosenbach,N.U.F.,Fair,D.A.,Miezin,F.M.,Cohen,A.L.,Wenger,K.K.,Dosenbach,R.
http://dx.doi.org/10.1016/j.tics.2008.01.006.
A.T....,&Petersen,S.E.(2007).Distinctbrainnetworksforadaptiveandstable
Anderson,J.R.,Qin,Y.,Jung,K.J.,&Carter,C.S.(2007).Information-processing
taskcontrolinhumans.ProceedingsoftheNationalAcademyofSciencesofthe
modules and their relative modality specificity. Cognitive Psychology, 54(3),
USA,104(26),11073–11078.http://dx.doi.org/10.1073/pnas.0704320104.
185–217.http://dx.doi.org/10.1016/j.cogpsych.2006.06.003.
Anderson,J.R.,&Schooler,L.(1991).Reflectionsoftheenvironmentinmemory. Forstmann,B.U.,Dutilh,G.,Brown,S.,Neumann,J.,vonCramon,D.Y.,Ridderinkhof,
PsychologicalScience,2(6). K.R.,etal.(2008).Striatumandpre-SMAfacilitatedecision-makingundertime
Borst,J.P.,&Anderson,J.R.(2013).Usingmodel-basedfunctionalMRItolocate pressure.ProceedingsoftheNationalAcademyofSciencesoftheUSA,105(45),
workingmemoryupdatesanddeclarativememoryretrievalsintheFronto- 17538–17542.http://dx.doi.org/10.1073/pnas.0805903105.
Parietalnetwork.ProceedingsoftheNationalAcademyofSciencesoftheUnited
Friston,K.J.,Ashburner,J.T.,Kiebel,S.J.,Nichols,T.E.,&Penny,W.D.(2007).
StatesofAmerica,110(5),1628–1633.
Statisticalparametricmapping.Theanalysisoffunctionalbrainimages.Academic
http://dx.doi.org/10.1073/pnas.1221572110.
Borst,J.P.,&Anderson,J.R.(2015).UsingthecognitivearchitectureACT-Rin Press.
combinationwithfMRIdata.InB.U.Forstmann,&E.-J.Wagenmakers(Eds.), Gläscher,J.P.,&O’Doherty,J.P.(2010).Model-basedapproachestoneuroimaging:
Model-basedcognitiveneuroscience.NewYork:Springer. combiningreinforcementlearningtheorywithfMRIdata.WileyInterdisciplinary
Borst,J.P.,Buwalda,T.A.,VanRijn,H.,&Taatgen,N.A.(2013).Avoidingtheproblem Reviews:CognitiveScience,1(4),501–510.http://dx.doi.org/10.1002/wcs.57.
statebottleneckbystrategicuseoftheenvironment.ActaPsychologica,144(2), Meyer, D. E., & Kieras, D. E. (1997). A computational theory of executive
373–379. cognitiveprocessesandmultiple-taskperformance:Part1.Basicmechanisms.
Borst,J.P.,Nijboer,M.,Taatgen,N.A.,VanRijn,H.,&Anderson,J.R.(2015).Using
PsychologicalReview,104(1),3–65.
data-drivenmodel-brainmappingstoconstrainformalmodelsofcognition.
PLoSOne,10(3),e0119673.http://dx.doi.org/10.1371/journal.pone.0119673. Newell,A.(1973).Youcan’tplay20questionswithnatureandwin:projective
Borst,J.P.,Taatgen,N.A.,Stocco,A.,&VanRijn,H.(2010).Theneuralcorrelates comments on the papers of this symposium. In W. G. Chase (Ed.), Visual
of problem states: Testing fMRI predictions of a computational model of informationprocessing(pp.283–308).NewYork:AcademicPress.
multitasking.PLoSOne,5(9),e12966.
Newell,A.(1990).Unifiedtheoriesofcognition.Cambridge,MA:HarvardUniversity
Borst,J.P.,Taatgen,N.A.,&VanRijn,H.(2011).Usingasymbolicprocessmodelas
Press.
inputformodel-basedfMRIAnalysis:Locatingtheneuralcorrelatesofproblem
statereplacements.NeuroImage,58(1),137–147. Nijboer,M.,Borst,J.P.,VanRijn,H.,&Taatgen,N.A.(2016).Contrastingsingle
http://dx.doi.org/10.1016/j.neuroimage.2011.05.084. and multi-component working-memory systems in dual tasking. Cognitive
Borst, J. P., Taatgen, N. A., & Van Rijn, H. (2015). What makes interruptions Psychology,86,1–26.http://dx.doi.org/10.1016/j.cogpsych.2016.01.003.
disruptive? A process-model account of the effects of the problem state O’Doherty,J.P.,Hampton,A.,&Kim,H.(2007).Model-basedfMRIanditsapplication
bottleneckontaskinterruptionandresumption.InProc.CHI.Seoul,Korea:ACM torewardlearninganddecisionmaking.AnnalsoftheNewYorkAcademyof
Press. Sciences,1104,35–53.http://dx.doi.org/10.1196/annals.1390.022.
Brett,M.,Anton,J.L.,Valabregue,R.,&Poline,J.B.(2002).Regionofinterestanalysis
usinganSPMtoolbox.Neuroimage,16(2). Pitt,M.A.,&Myung,I.(2002).Whenagoodfitcanbebad.TrendsinCognitive
Cabeza,R.,Dolcos,F.,Prince,S.E.,Rice,H.J.,Weissman,D.H.,&Nyberg,L.(2003). Sciences,6(10),421–425.
Attention-relatedactivityduringepisodicmemoryretrieval:across-function Roberts,S.,&Pashler,H.(2000).Howpersuasiveisagoodfit?Acommentontheory
fMRIstudy.Neuropsychologia,41(3),390–399. testing.PsychologicalReview,107(2),358–367.
Chen,M.C.,Anderson,J.R.,&Sohn,M.H.(2001).Whatcanamousecursortell
usmore?:correlationofeye/mousemovementsonwebbrowsing.InHuman Salvucci,D.D.(2006).Modelingdriverbehaviorinacognitivearchitecture.Human
factorsincomputingsystems,CHI’01.(pp.281–282).Seattle,Washington:ACM. Factors,48(2),362–380.
Cole,M.W.,&Schneider,W.(2007).Thecognitivecontrolnetwork:Integrated Schneider, D. W., & Anderson, J. R. (2012). Modeling fan effects on the
cortical regions with dissociable functions. NeuroImage, 37(1), 343–360. timecourseofassociativerecognition.CognitivePsychology,64(3),127–160.
http://dx.doi.org/10.1016/j.neuroimage.2007.03.071. http://dx.doi.org/10.1016/j.cogpsych.2011.11.001.

J.P.Borst,J.R.Anderson/JournalofMathematicalPsychology76(2017)94–103 103
Shiffrin,R.M.,Lee,M.D.,Kim,W.,&Wagenmakers,E.-J.(2008).Asurveyofmodel Turner,B.M.,vanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
evaluationapproacheswithatutorialonhierarchicalBayesianmethods.Cogni- stractionsthroughneuroimaging:theneuraldriftdiffusionmodel.Psychological
tiveScience,32(8),1248–1284.http://dx.doi.org/10.1080/03640210802414826. Review,122(2),312–336.http://dx.doi.org/10.1037/a0038894.
Spreng, R. N., Stevens, W. D., Chamberlain, J. P., Gilmore, A. W., & Schacter, vanMaanen,L.,Brown,S.D.,Eichele,T.,Wagenmakers,E.-J.,Ho,T.,Serences,J.,etal.
D. L. (2010). Default network activity, coupled with the frontoparietal (2011).Neuralcorrelatesoftrial-to-trialfluctuationsinresponsecaution.The
controlnetwork,supportsgoal-directedcognition.NeuroImage,53(1),303–317. JournalofNeuroscience,31(48),17488–17495.
http://dx.doi.org/10.1016/j.neuroimage.2010.06.016. http://dx.doi.org/10.1523/jneurosci.2924-11.2011.
Stocco,A.,&Anderson,J.R.(2008).Endogenouscontrolandtaskrepresentation:
anfMRIstudyinalgebraicproblem-solving.JournalofCognitiveNeuroscience, Wintermute, S., Betts, S., Ferris, J. L., Fincham, J. M., & Anderson, J. R.
20(7),1300–1314.http://dx.doi.org/10.1162/jocn.2008.20089. (2012). Brain networks supporting execution of mathematical skills versus
Taatgen, N. A. (2013). The nature and transfer of cognitive skills. Psychological acquisition of new mathematical competence. PLoS One, 7(12), e50154.
Review,120,439–471. http://dx.doi.org/10.1371/journal.pone.0050154.