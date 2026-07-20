JournalofMathematicalPsychology76(2017)212–235
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Model-based functional neuroimaging using dynamic neural fields:
An integrative cognitive neuroscience approach
SobanawartinyWijeakumara,∗ ,JosephP.Ambroseb,∗ ,JohnP.Spencera,∗ ,RodicaCurtuc
aUniversityofEastAnglia,SchoolofPsychology,NorwichNR47TJ,UnitedKingdom
bUniversityofIowa,DepartmentofPsychologyandDeltaCenter,IowaCity52242,IA,USA
cUniversityofIowa,DepartmentofMathematicsandDeltaCenter,IowaCity52242,IA,USA
h i g h l i g h t s
• InteractivecognitiveneuroscienceapproachisdemonstratedusingDFT.
• DNFmodelsfittedtrendsinbehavioralandneuraldatatoresponseselectiontask.
• NeuralpopulationdynamicsinDNFmodelweremappedtoparticularbrainregions.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Afundamentalchallengeincognitiveneuroscienceistodeveloptheoreticalframeworksthateffectively
Availableonline21December2016 spanthegapbetweenbrainandbehavior,betweenneuroscienceandpsychology.Here,weattemptto
bridgethisdividebyformalizinganintegrativecognitiveneuroscienceapproachusingdynamicfield
Keywords:
theory(DFT).WebeginbyprovidinganoverviewofhowDFTseekstounderstandtheneuralpopulation
Dynamicfieldtheorymodeling
dynamics that underlie cognitive processes through previous applications and comparisons to other
Integrativecognitiveneuroscience
modelingapproaches.Wethenusepreviouslypublishedbehavioralandneuraldatafromaresponse
Responseselection
Functionalmagneticresonanceimaging selection Go/Nogo task as a case study for model simulations. Results from this study served as the
‘standard’forcomparisonswithamodel-basedfMRIapproachusingdynamicneuralfields(DNF).The
tutorialexplainstherationaleandhypothesesinvolvedintheprocessofcreatingtheDNFarchitecture
and fitting model parameters. Two DNF models, with similar structure and parameter sets, are then
compared.Bothmodelseffectivelysimulatedreactiontimesfromthetaskaswevariedthenumberof
stimulus–responsemappingsandtheproportionofGotrials.Next,wedirectlysimulatedhemodynamic
predictionsfromtheneuralactivationpatternsfromeachmodel.Thesepredictionsweretestedusing
generallinearmodels(GLMs).ResultsshowedthattheDNFmodelthatwascreatedbytuningparameters
tocapturesimultaneouslytrendsinneuralactivationandbehavioraldataquantitativelyoutperformeda
StandardGLManalysisofthesamedataset.Further,byusingtheGLMresultstoassignfunctionalrolesto
particularclustersinthebrain,weillustratehowDNFmodelsshednewlightontheneuralpopulations’
dynamicswithinparticularbrainregions.Thus,thepresentstudyillustrateshowaninteractivecognitive
neurosciencemodelcanbeusedinpracticetobridgethegapbetweenbrainandbehavior.
©2016ElsevierInc.Allrightsreserved.
1. Introduction theories (Gerstner, Sprekeler, & Deco, 2012). But what would a
good theory of brain function look like? Addressing this question
Although great strides have been made in understanding the requirestheoriesthatbridgethedisparatescientificlanguagesof
brain using data-driven methods (Smith, Fox, Miller, Glahn, Fox, neuroscienceandpsychology.
& Mackay, 2009), human neuroscience will need sophisticated Turner, Forstmann, Love, Palmeri, and Van Maanen (2016)
describedthreecategoriesofapproachestothisissueusingmodel-
based cognitive neuroscience that bridge the gap between brain
∗ andbehaviorbybringingtogetherfMRIdataandcognitivemodels
Correspondingauthors.
(Turneretal.,2016).Thefirstapproachusesneuraldatatoguide
E-mailaddresses:S.Wijeakumar@uea.ac.uk(S.Wijeakumar),
j.spencer@uea.ac.uk(J.P.Spencer). and inform a behavioral model, that is, a model that mimics
http://dx.doi.org/10.1016/j.jmp.2016.11.002
0022-2496/©2016ElsevierInc.Allrightsreserved.

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 213
features of responses such as reaction times and accuracy. One Spencer,2013).Theyfirstconstructedadynamicneuralfield(DNF)
example of this approach is the Leaky Competing Accumulator modelofthedual-taskparadigmreportedbyDuxandcolleagues
model by Usher and McClelland (Usher & McClelland, 2001). (Duxetal.,2009).Themodelquantitativelyfitacomplexpattern
|           |             |       |     |          |               |     |       | of reaction | time | changes | over learning, |     | including | the reduction |     |
| --------- | ----------- | ----- | --- | -------- | ------------- | --- | ----- | ----------- | ---- | ------- | -------------- | --- | --------- | ------------- | --- |
| This is a | mechanistic | model | for | evidence | accumulation, |     | which |             |      |         |                |     |           |               |     |
incorporateswell-knownpropertiesofneuronalensemblessuch of dual-task costs over learning to single task levels. These
as leakage and lateral inhibition. The model provides a good fit researchersthengeneratedaLFPmeasurefromeachcomponent
forarangeofbehavioraldata,forexample,time–accuracycurves of the neural model and convolved the LFPs with an impulse
andtheeffectsofthenumberofalternativesonchoiceresponse responsefunctiontogenerateBOLDresponsesfromthemodel.The
times.Unfortunately,asremarkedbyTurneretal.,thismechanistic DNF model captured key fMRI results from Dux et al., including
approachstopsshortofestablishinganydirectconnectiontothe the reduction of the amplitude of the hemodynamic response
dynamicsofparticularneuralcircuitsorbrainareas. ininferiorfrontaljunctionindual-taskconditionsoverlearning.
The second type of approach uses a behavioral model and Moreover,Bussetal.contrastedcompetingpredictionsoftheDNF
applies it to the prediction of neural data. One example of this model and ACT-R, showing that changes in hemodynamics over
approach is Rescorla and Wagner’s (1972) model of learning learningpredictedbytheDNFmodelmatchedfMRIresultsfrom
Duxetal.,whilepredictionsfromACT-Rdidnot.
| conditioned | responses. | In  | this model, | thevalue |     | of a | conditioned |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | ----------- | -------- | --- | ---- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
ItisimportanttohighlightseveralkeypointsachievedbyBuss
stimulusisupdatedoversuccessivetrialsaccordingtoalearning
etal.(2013).First,theDNFmodelsimulatedneuraldynamicsin
rateparameter.Themodelproducestrial-by-trialestimatesofthe
realtime.Thedynamicscreatedrobust‘peaks’ofactivationthat
| error between | the      | conditioned |     | and unconditioned |        |        | stimuli. This |               |        |               |     |           |     |            |     |
| ------------- | -------- | ----------- | --- | ----------------- | ------ | ------ | ------------- | ------------- | ------ | ------------- | --- | --------- | --- | ---------- | --- |
|               |          |             |     |                   |        |        |               | were directly | linked | to behavioral |     | responses | by  | the model, | and |
| measure       | can then | be used     | in  | general           | linear | models | to detect     |               |        |               |     |           |     |            |     |
theseresponsesquantitativelycapturedacomplexpatternofre-
| patterns | matching | the model | predictions |     | within | fMRI | data. The |     |     |     |     |     |     |     |     |
| -------- | -------- | --------- | ----------- | --- | ------ | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
actiontimesoverlearning.Second,thesameneuraldynamicsthat
methodpotentiallyallowsonetoidentifyneuralprocessesthatare
quantitativelyfitbehavioralsosimulatedobservedhemodynamics
| not directly | measurable |     | through | behavioral | results |     | (Davis, Love, |     |     |     |     |     |     |     |     |
| ------------ | ---------- | --- | ------- | ---------- | ------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
& Preston, 2012; Mack, Preston, & Love, 2013; Palmeri, Schall, measuredwithfMRI.Finally,Bussetal.demonstratedthespeci-
& Logan, 2015). However, a drawback of this model-based fMRI ficityofthesefindingsbycontrastedpredictionsoftwotheories.
Thus,theirworkconstitutesanotableexampleofanintegrative
| approach | is that | it does | not explain | cognitive |     | states | encoded by |     |     |     |     |     |     |     |     |
| -------- | ------- | ------- | ----------- | --------- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
cognitiveneuroscienceapproachusinganeuralprocessmodelthat
patternsofactivationdistributedovermultiplevoxelsinthebrain.
capitalizesonconstraintsregardinghowbrainswork.
Thelast,andmostdifficultapproachisanintegrativecognitive
|              |            |              |            |             |                |          |             | The current    |             | paper builds | on           | the above   | example,   | by      | formal- |
| ------------ | ---------- | ------------ | ---------- | ----------- | -------------- | -------- | ----------- | -------------- | ----------- | ------------ | ------------ | ----------- | ---------- | ------- | ------- |
| neuroscience | approach   |              | where a    | model       | simultaneously |          | predicts    |                |             |              |              |             |            |         |         |
|              |            |              |            |             |                |          |             | izing an       | integrative | cognitive    | neuroscience |             | approach   | using   | dy-     |
| behavioral   | and        | neural       | data. That | is,         | the model      | explains | what        |                |             |              |              |             |            |         |         |
|              |            |              |            |             |                |          |             | namic neural   | fields.     | Our          | paper is     | tutorial    | in nature, | walking | the     |
| the brain    | is doing   | in real-time |            | to generate |                | specific | patterns of |                |             |              |              |             |            |         |         |
|              |            |              |            |             |                |          |             | reader through |             | each step    | of this      | model-based | cognitive  |         | neuro-  |
| fMRI and     | behavioral | data.        | Turner     | et al.      | acknowledge    |          | that there  |                |             |              |              |             |            |         |         |
scienceframework.WeextendtheworkofBussetal.(2013)by
| are relatively | few | examples | in  | this category. |     | For instance, | they |     |     |     |     |     |     |     |     |
| -------------- | --- | -------- | --- | -------------- | --- | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
(1)formalizingseveralstepsregardingthecalculationofLFPsfrom
| highlight | recent     | papers  | that use | cognitive          |     | architectures | such       |         |        |            |                |     |         |              |     |
| --------- | ---------- | ------- | -------- | ------------------ | --- | ------------- | ---------- | ------- | ------ | ---------- | -------------- | --- | ------- | ------------ | --- |
|           |            |         |          |                    |     |               |            | dynamic | neural | fields and | the generation |     | of BOLD | predictions; |     |
| as ACT-R  | (‘Adaptive | Control | of       | Thought-Rational’) |     |               | to capture |         |        |            |                |     |         |              |     |
(2)addingnewmethodstoquantitativelyevaluateBOLDpredic-
| simultaneously | fMRI | and | behavioral | data | (Anderson, |     | Matessa, & |     |     |     |     |     |     |     |     |
| -------------- | ---- | --- | ---------- | ---- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
tionsfromdynamicneuralfieldmodelsusinggenerallinearmod-
Lebiere, 1997; Borst & Anderson, 2013; Borst, Nijboer, Taatgen, els (GLM), inspired by other model-based fMRI approaches; and
VanRijn,&Anderson,2015).Althoughweagreethatthisapproach (3)addingnewmethodstoidentifymodel-basedfunctionalnet-
has immense potential, this is a relatively limited example of worksfromgroup-levelGLMresults.Thesemethodsallowforef-
anintegrativecognitiveneuroscienceapproachbecauseACT-Ris fectively identifying where particular neural patterns live in the
| not a neural | process | model. | Thus, | ACT-R | does | not | capitalize on |     |     |     |     |     |     |     |     |
| ------------ | ------- | ------ | ----- | ----- | ---- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
brain,aswellasspecifyingtheirfunctionalroles.
constraintsregardinghowrealbrainsactuallywork.
|                |     |          |      |      |            |     |             | The | paper proceeds |     | as follows. | We  | begin | with | a brief |
| -------------- | --- | -------- | ---- | ---- | ---------- | --- | ----------- | --- | -------------- | --- | ----------- | --- | ----- | ---- | ------- |
| An alternative |     | approach | that | does | capitalize | on  | neural con- |     |                |     |             |     |       |      |         |
introductiontodynamicfieldtheory.Thisplacesourmodel-based
straintswasproposedbyDeco,Rolls,andHorwitz(2004).These
|     |     |     |     |     |     |     |     | approach | within | a broader | context | for | readers | who might | be  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------ | --------- | ------- | --- | ------- | --------- | --- |
researchersusedintegrate-and-fireattractornetworkstosimulate
|     |     |     |     |     |     |     |     | less familiar | with | this theoretical |     | approach. | Next, | we introduce |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | ---------------- | --- | --------- | ----- | ------------ | --- |
neuralactivityfroma‘where-and-what’task.Themodelincludes
theparticularcasestudywewillusethroughoutthepaper,that
severalpopulationsofsimulatedneuronstoreflectnetworkstuned
|     |     |     |     |     |     |     |     | is, the particular |     | behavioral | and | fMRI dataset | that | serves | as the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ---------- | --- | ------------ | ---- | ------ | ------ |
tospecificobjects,positions,orcombinationsthereof.Theauthors basis for the tutorial. We then discuss the DNF model that we
then define a local field potential (LFP) measure from each neu- usedtocapturesimultaneouslybehavioralandneuraldatafrom
ralpopulationbyaveragingthesynapticflowateachtimestep.To this study, explaining where this model comes from and how
generateaBOLDresponse,theyconvolvedtheLFPmeasurewithan we approached the simulation case study. The presentation will
impulseresponsefunction.Althoughoneversionofthemodelwas highlight key issues that theoreticians face when adopting an
abletoapproximatesingleneuronrecordingsfromapriorstudy, integrative cognitive neuroscience approach. Next, we present
aswellasameasuredfMRIpatternindorsolateralprefrontalcor- behavioralfitsofthedataanddiscussstrengthsandlimitationsof
tex,otherfMRIpatternsfromtheventrolateralprefrontalcortex theDNFmodelatthislevelofanalysis.
werenotmodeled.Moreover,comparisonstofMRIdataweremade Afterconsideringthebehavioraldata,weintroduceastep-by-
qualitativelyviavisualinspection.Noattemptwasmadetoquan- stepguidetogeneratinghemodynamicpredictionsfromdynamic
|            |            |           |          |     |            |      |           | neural field | models. | We  | then | discuss | how to | evaluate | these |
| ---------- | ---------- | --------- | -------- | --- | ---------- | ---- | --------- | ------------ | ------- | --- | ---- | ------- | ------ | -------- | ----- |
| titatively | relate the | measures. | Finally, |     | behavioral | data | from this |              |         |     |      |         |        |          |       |
studywerenotacentralfocus.Suchissuesarerelativelycommon predictionsusinggenerallinearmodeling(GLM).Wefirstevaluate
whenmodelingreliesonbiophysicalneuralnetworksduetothe the model predictions at the individual level. We then move
immensecomputationalchallengesofsimulatingsuchnetworks. to the group level, showing how our approach can be used
Appropriatepartitioningoftheparameterspaceandestimationof to identify model-based functional networks. To evaluate these
modelparametersare,ingeneral,difficultstepsofthisapproach networks, we compare our approach to standard fMRI analyses,
(seeAnderson,2012;Turneretal.,2016). highlighting examples where the DNF model sheds interesting
Inspired by this work, Buss, Wifall, Hazeltine, and Spencer lightonthefunctionalrolesofparticularbrainregions.Thetutorial
(2013)adaptedthisapproachtosimultaneouslymodelbehavioral concludeswithageneralevaluationofourmodel-basedapproach,
andfMRIdatafromadual-taskparadigm(Buss,Wifall,Hazeltine,& highlightingstrengths,weaknesses,andfuturedirections.

214 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
2. Overviewofdynamicfieldtheory withinacorticalfield,butpeaksgoawaywheninputisremoved;
andaself-sustainingorworkingmemorystatewhereactivation
The present report introduces a tutorial on an integrative peaksremainstableevenintheabsenceofinput.Movementinto
model-based fMRI approach using Dynamic Field Theory (DFT). and out of these states is assembled in real-time depending on
Thus, for clarity, before explaining the integrative cognitive a variety of factors including inputs to a field. Critically, though,
neuroscienceapproach,westartbygivingabriefintroductionto activationpatternscan‘‘riseabove’’thecurrentinputpatternvia
DFT.ReadersarereferredtotheDFTResearchGroup(2015)fora recurrent interactions: activation can be in a stable ‘‘on’’ state
thoroughtreatmentoftheseideas. wheresubsequentinputsaresuppressed.Thatsaid,the‘‘on’’state
DFT grew out of the principles and concepts of dynamical is still open to change: in the presence of continued input, the
systems (Schöner, Spencer, & The DFT Research Group, 2015) network might ‘‘update’’ its decision to focus on one item over
theory initially explored in the ‘motor approach’ pioneered by another. This points toward flexibility—how activation patterns
Gregor Schöner, Esther Thelen, Scott Kelso, and Michael Turvey can go smoothly and autonomously from one stable state to
(Kelso, Scholz, & Schoner, 1988; Schöner & Kelso, 1988; Turvey, another.
1995).Thegoalwastodevelopaformal,neurally-groundedtheory Todate,severalstrengthsofDFTareevident.First,DFTprovides
thatcouldbringtheconceptsofdynamicalsystemstheorytobear a predictive language to understand both brain and behavior.
onissuesincognitionandcognitivedevelopment(fordiscussion, DFThasbeenusedtotestspecificpredictionsaboutearlyvisual
seeSpencer&Schoner,2003).DFTwasinitiallyappliedtoissues processing, attention, working memory, response selection, and
closely aligned with the cognitive aspects of motor systems spatial cognition at behavioral and brain levels using multiple
such as motor planning for arm and eye movements (Erlhagen neuroscience technologies (Johnson et al., 2009; Markounikau,
& Schöner, 2002; Kopecz & Schöner, 1995). Subsequent work Igel, Grinvald, & Jancke, 2010; Schneegans et al., 2014; Schutte
extendedDFT,capturingawidearrayofphenomenaintheareaof et al., 2003). Second, DFT scales up. Across several papers, we
spatially-groundedcognition,frominfantperseverativereaching havedemonstrated,forinstance,that‘local’theoriesofattention,
(Smith,Thelen,Titzer,&McLin,1999;Thelen,Schöner,Scheier,& working memory, and response selection can be integrated in a
Smith, 2001) to spatial category biases to changes in the metric large-scaleneuralmodelthatexplainsandpredictshowhumans
precisionofspatialworkingmemoryfromchildhoodtoadulthood represent objects in a visual scene—see Schöner et al. (2015).
(Schutte,Spencer,&Schöner,2003;Simmering,Peterson,Darling, Third,DFTiswellpositionedtobridgethegapbetweenbrainand
& Spencer, 2008). In the last decade, DFT has been extended behavior,simultaneouslygeneratingreal-timeneuralpopulation
into a host of other domains including visual working memory dynamicsandresponsesthatmimicbehavior,ofteninquantitative
[VWM] (Johnson, Hollingworth, & Luck, 2008; Johnson, Spencer, detail(Bussetal.,2013;Erlhagen&Schöner,2002).
Luck, & Schöner, 2009; Schneegans, Spencer, Schöner, Hwang, & TheneuralgroundingofDFThasbeeninvestigatedusingboth
Hollingworth, 2014), retinal remapping (Schneegans & Schöner, multi-unitneurophysiology(Bastian,Riehle,Erlhagen,&Schöner,
2012), preferential looking and visual habituation (Perone & 1998; Erlhagen et al., 1999) and voltage-sensitive dye imaging
Spencer,2013;Perone,Spencer,&Schöner,2007),spatiallanguage (Markounikauetal.,2010).Datafromthesestudiesdemonstrate
(Lipinski,Spencer,&Samuelson,2010),wordlearning(Samuelson, that DFT can capture the details of neural population activation
Jenkins, & Spencer, 2015), executive function (Buss & Spencer, in the brain and generate novel, neural predictions (Bastian,
2008), and autonomous behavioral organization in cognitive Schöner, & Riehle, 2003; Markounikau et al., 2010). Thus, the
robotics(Sandamirskaya&Schöner,2010). neural grounding of DFT extends beyond mere analogy. Rather,
The dynamic field framework was initially developed to DFTimplementsasetofformalhypothesesabouthowthebrain
understand brain function at the level of neural population worksthatcanbedirectlytestedusingneurosciencemethods.It
dynamics. Evidence suggests that local neural populations move wasthesuccessofthisframeworkatcapturingthedetailsofneural
intoandoutofattractorstates,reliablepatternsofactivationthat populationdynamicsinthebrainthatencouragedustoconsider
theneuralpopulationmaintainsinthecontextofparticularinputs. themappingbetweenneuralpopulationdynamicsandtheBOLD
Forinstance,whenpresentedwithvisualinput,neuralpopulations signalmeasuredwithfMRI.Theintegrativecognitiveneuroscience
invisualcortexcreatestable‘peaks’ofactivationthatindicatethat approachdetailedhereisacriticalstepinthisnewdirection.
somethingisontheleftsideoftheretina(Erlhagen,Bastian,Jancke,
Riehle,&Schöner,1999;Markounikau,Igel,&Jancke,2008).These 3. Introductiontothecasestudy
localdecisions–peaks–thenshareactivationwithotherneural
populations – other peaks – creating a macro-scale brain state. Toillustratethemodel-basedapproachtofMRIusingDFT,we
Thinking,accordingtoDFT,isthemovementintoandoutofthese have to select a specific case study. This anchors the modeling
states.Behavingistheconnectionofthesestatestosensorimotor approach to a specific task, a specific set of behaviors, and a
systems. Learning is the refinement of these patterns via the specificfMRIdataset. Here,weuseascasestudytheneuraland
constructionoflocalizedmemorytracesandconnectivitybetween behavioral dynamics that underlie response selection. Response
fields. Development is the shaping of neural activation patterns selection has been studied using DFT for almost two decades
step-by-stepthroughhours,days,weeks,andyearsofgeneralized at both behavioral (Christopoulos, Bonaiuto, & Andersen, 2015;
experience. Erlhagen & Schöner, 2002; Klaes, Schneegans, Schöner, & Gail,
Formally,dynamicneuralfieldmodelsareinaclassofbi-stable 2012; McDowell, Jeka, Schöner, & Hatfield, 1998, 2002; Schutte
neuralnetworksfirstdevelopedbyAmari(1977),andthenstudied &Spencer,2007)andneurallevels(Bastianetal.,1998;Erlhagen
theoreticallyandcomputationallybymanyresearchgroupsover et al., 1999; McDowell et al., 2002). Thus, there is a rich history
lasttwodecades(Bressloff,2001;Coombes&Owen,2005;Curtu tobuildon.Furthermore,thelastdecadehasseenanexplosionof
&Ermentrout,2004;Ermentrout&Kleinfeld,2001;Jirsa&Haken, researchexaminingthebehavioralandneuralbasesforresponse
1997; Laing & Chow, 2001; Wilson & Cowan, 1973; Wong & selectionandinhibitionusingfMRI.Thisstems,inpart,fromthe
Wang,2006).Activationinthesenetworks–called‘corticalfields’ clinical relevance of this topic: poor performance on response
– is distributed over continuous dimensions—space, movement selectiontaskshasbeenlinkedtoperformancedeficitsinatypical
direction, color, and so on. Importantly, patterns of activation populations (Kaladjian, Jeanningros, Azorin, Anton, & Mazzola-
can live in different ‘‘attractor’’ states: a resting state; an input- Pomietto,2011;Monterosso,Aron,Cordova,Xu,&London,2005;
driven state where input forms stabilized ‘‘peaks’’ of activation Pliszka,Liotti,&Woldorff,2000).

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 215
trials.Inthe75%condition,75%ofthetrialswereGotrialsand25%
ofthetrialswereNogotrials.
FortheLoadmanipulation,50%ofthetrialswereGotrialsand
|     |     |     |     |     |     |     | the rest    | were Nogo  | trials. | In the   | Load          | 2 condition, |     | one stimulus |      |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | ------- | -------- | ------------- | ------------ | --- | ------------ | ---- |
|     |     |     |     |     |     |     | (color) was | associated |         | with     | a Go response |              | and | another      | with |
|     |     |     |     |     |     |     | the Nogo    | response.  | In      | the Load | 4 condition,  |              | two | stimuli      | were |
associatedwithaGostimulusandtwootherstimuliwithaNogo
|     |     |     |     |     |     |     | response.    | In the       | Load 6     | condition, | three   | stimuli  | were          | associated |        |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | ---------- | ---------- | ------- | -------- | ------------- | ---------- | ------ |
|     |     |     |     |     |     |     | with the     | Go response  |            | and three  | stimuli | with     | a Nogo        | response.  |        |
|     |     |     |     |     |     |     | Participants | completed    |            | five runs  | in      | the fMRI | experiment:   |            | Load   |
|     |     |     |     |     |     |     | 2, Load 4    | (also called | Proportion |            | 50),    | Load     | 6, Proportion |            | 25 and |
Proportion75.Eachrunhadatotalof144trials.Theorderofthe
runswasrandomized.
fMRIdatawerecollectedusinga3TSiemensTIMTriomagnetic
Fig.1. ExperimentaldesignfortheGnGtask.
|     |     |     |     |     |     |     | resonance | imaging | system | with | a 12-channel |     | head | coil. An | MP- |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | ------ | ---- | ------------ | --- | ---- | -------- | --- |
In a recent paper (Wijeakumar et al., 2015), we contributed RAGEsequencewasusedtocollectanatomicalT1-weightedvol-
to this fMRI literature by examining whether response selection umes. Functional BOLD imaging was acquired using an axial 2D
andinhibitionareasinthebrainareactiveprimarilyoninhibitory echo-planargradientechosequencewiththefollowingparame-
ters:TE=30ms,TR=2000ms,flipangle=70°,FOV=240Å∼
trialsassomeresearchershaveclaimed(Aron,Robbins,&Poldrack,
|     |     |     |     |     |     |     | 240mm,matrix |     | = 64Å | ∼ 64,slicethickness/gap=4.0/1.0mm, |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | ---------------------------------- | --- | --- | --- | --- | --- |
2014),or,alternatively,whetherresponseselectionandinhibition
andbandwidth=1920Hz/pixel.
areasareactivewhensalienteventsoccur,regardlessofwhether
|              |                 |            |          |                     |     |                | The task | was               | presented | to         | the participant |        | inside    | the scanner |         |
| ------------ | --------------- | ---------- | -------- | ------------------- | --- | -------------- | -------- | ----------------- | --------- | ---------- | --------------- | ------ | --------- | ----------- | ------- |
| these events | require         | inhibition | per      | se (Erika-Florence, |     | Leech, &       |          |                   |           |            |                 |        |           |             |         |
|              |                 |            |          |                     |     |                | through  | a high-resolution |           | projection |                 | system | connected |             | to a PC |
| Hampshire,   | 2014; Hampshire |            | & Sharp, | 2015).              | To  | contrast these |          |                   |           |            |                 |        |           |             |         |
usingE-primesoftware.Thetimingofthestimulibeingpresented
| views, we | had participants |     | complete | a set | of classic | inhibitory |     |     |     |     |     |     |     |     |     |
| --------- | ---------------- | --- | -------- | ----- | ---------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
controltasksinanMRIscanner.Wevariedwhethereventswere was synchronized to the MRI scanner’s trigger pulse. Head
excitatory (i.e., required a motor response) or inhibitory, and movementwaspreventedbyinsertingfoampaddingbetweenthe
whethereventswerefrequentorinfrequent.Wewereparticularly participants’headsandtheheadcoil.Participants’responseswere
interested in the brain response on infrequent, excitatory trials. obtained through a manipulandam strapped to the participants’
| Theinhibitorynetworkviewsuggeststhatkeyareasofafronto- |     |     |     |     |     |     | hand. |     |     |     |     |     |     |     |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
cortical–striatalnetworkshouldshowaweakresponseonthese DatawereanalyzedusingAnalysisofFunctionalNeuroImages
trials because no inhibition is required. The salience network (AFNI) software (http://afni.nimh.nih.gov/afni). DICOM images
|     |     |     |     |     |     |     | were converted |     | to NIFTI | images. | Voxels | containing |     | non-brain |     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | -------- | ------- | ------ | ---------- | --- | --------- | --- |
viewsuggeststheopposite—thatthereshouldbearobustfronto-
|                   |         |          |     |         |            |        | tissue were | stripped |     | from | the T1 | structural | image. |     | The T1 |
| ----------------- | ------- | -------- | --- | ------- | ---------- | ------ | ----------- | -------- | --- | ---- | ------ | ---------- | ------ | --- | ------ |
| cortical–striatal | network | response |     | because | infrequent | events |             |          |     |      |        |            |        |     |        |
structuralimagewasalignedtotheTalairachspace.Then,EPIdata
standoutassalient.
wastransformedtoalignwiththeT1structuralscaninthesubject-
| We used | the data | from | Wijeakumar | et  | al. (2015) | as our case |     |     |     |     |     |     |     |     |     |
| ------- | -------- | ---- | ---------- | --- | ---------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
studyinthepresentreport.Wedothisfortworeasons.First,this space. Transformation matrices across both these steps were
is a convenient choice because we have the full dataset, we are concatenatedandappliedtotheEPIdatatomoveitfromsubject-
aware of all the processing details, and so on. Second, although space to Talaraich space. Six parameters for head movement
, ,
therearenumerousotherstudieswecouldhavepicked,thisone wereestimatedX Y Z,pitch,roll,andyawdirections)foruseas
hassomeuniquefeatures.Mostnotably,thestudyofWijeakumar regressors to account for variance in the BOLD signal associated
etal.hasparametricallymanipulatedseveralfactorsinthesame withmotion.Spatialsmoothingwasperformedonthefunctional
task.Thisisgoodfoddertoprobethepotentialofourmodel-based datausingaGaussianfunctionof8mmfull-widthhalf-maximum.
approachbecausethereisalotofsystematicpatterninginthedata Results showed a robust neural response in key areas of the
fronto-cortical–striatalnetworkoninfrequenttrialsregardlessof
tocapture.
|            |                 |     |            |         |         |             | the need   | for inhibition       |     | (Wijeakumar |      | et al.,  | 2015).    | Interestingly, |     |
| ---------- | --------------- | --- | ---------- | ------- | ------- | ----------- | ---------- | -------------------- | --- | ----------- | ---- | -------- | --------- | -------------- | --- |
| In the     | present report, |     | we focus   | on data | from    | one of the  |            |                      |     |             |      |          |           |                |     |
|            |                 |     |            |         |         |             | the number | of stimulus–response |     |             | (SR) | mappings | modulated |                | the |
| tasks from | Wijeakumar      | et  | al. (2015) | — a     | Go/Nogo | (GnG) task. |            |                      |     |             |      |          |           |                |     |
Participantswereaskedtopressabutton(Go)whentheysawsome neural signal across multiple brain areas, with a reduction in
stimuliandwithhold(Nogo)theirresponsewhenanothersetof the BOLD signal as the number of SR mappings increased. We
stimuliwerepresented.Stimulivariedincolorbutnotinshape. suggested that this might reflect competition among associative
Go colors were separated from Nogo colors by 60° in a uniform memoriesoftheSRmappingsastheSRloadincreased,consistent
huespacesuchthatdirectlyadjacentcolorswereassociatedwith with recent proposals (Cisek, 2012) and modeling work by
differentresponsetypes. Erlhagenandcolleagues(Erlhagen&Schöner,2002).
Eachtrialstartedwithafixationcrosspresentedatthecenter In the next section, we present an overview of a dynamic
ofthescreenfor2500ms,followedbythestimuluspresentationat neural field model designed to capture both the behavioral and
thecenterofthescreenfor1500ms(seeFig.1).Theparticipants neural dynamics that underlie performance in this study. Note
thatweusethemodelprimarilyinatutorialfashion—toillustrate
wereadvisedtorespondtothevisualstimuliasfastaspossible.Ifa
responsewasnotdetectedontheGotrials,thenamessagesaying the model-based fMRI approach using dynamic neural fields.
‘NoResponseDetected’waspresentedonthescreenfor250ms. Critically, we make no claims that this is an optimal model of
Inter-trialintervalswerejitteredbetween1000,2500or3500ms responseselection.Thereareothermorecomprehensivemodels
presentedon50%,25%or25%ofthetrialsrespectively. of inhibitory control in the literature. For instance, Wiecki and
Twoparametricmanipulationswerecarriedout—aProportion Frank’smodelofresponseinhibitionunifiesmanyfindingsfrom
manipulation and a Load manipulation. For the Proportion the inhibitory control literature and has simulated key aspects
manipulation(atLoad4),thenumberofGoandNogotrialswere of neural data from both neurophysiology and evoked-response
variedasfollows.Inthe25%condition,25%ofthetrialswereGo potentials(Wiecki&Frank,2013).Wethinkourmodelhassome
trialsand75%ofthetrialswereNogotrials.Inthe50%condition, interesting strengths relative to Wiecki and Frank’s model that
50% of the trials were Go trials and 50% of the trials were Nogo we highlight below, but it also has some interesting limitations

216 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
thatwealsohighlight.Thesestrengthsandlimitationsareuseful directly to a higher-level processing area – often dominated the
in a tutorial style paper like this to illustrate the range of neural activation patterns, thereby dominating the model-based
issuesonemustconsiderwhenpursuinganintegrativecognitive MRIsignalsaswell.Thissuggeststhatitisimportanttoembedthe
neurosciencemodel. neuralprocessesofinterestwithinafullyneuralsystemifyouwant
tocaptureneuraldynamicsinareasonableway.Concretely,this
4. Adynamicneuralfieldmodelofresponseselection meansthatwehadapriorireasonsforsimulatingearlyperceptual
andattentionalprocessesinthemodel,eventhoughmostmodels
A key question one must ask when modeling even the most donotdothisintheinterestofsimplicity.
basicoftasksiswhatperceptual,cognitive,andmotorprocesses
oneshouldtrytocaptureinthemodelandwhataspectsshouldbe 4.1. Conceptualoverviewandmodelarchitecture
leftoutintheinterestofsimplicity.Inmathematicalpsychology,
such issues are central given that model simplicity versus With that background in mind, Fig. 2 shows the architecture
complexity–oftenindexedbythenumberoffreeparameters–is of the model. This model is an integration of several models
akeydimensionalongwhichmodelsarecompared.TheGnGtask developed to simulate findings from VWM tasks (Johnson et al.,
isrelativelysimple;thus,wecanarticulatethesetofpossibilities. 2009; Schneegans et al., 2014; Schöner et al., 2015), consistent
One could consider modeling the following: (1) the early visual withourgoalofaskingwhetheramodelofVWMcangeneralize
processesthatperceiveandencodecolorspresentedinthevisual toaresponseselectiontask.Wedescribethearchitectureindetail
field; (2) the attentional processes that selectively attend to the below, pointing out links to prior work to justify why we have
presentedcolor;(3)thememoryandvisualcomparisonprocesses usedthisparticulararchitecturehere.Notethateachelementin
thatidentifywhetherthepresentedcolorisfromtheGoorNogo Fig.2isadynamicneuralfield.Weprovidethefullmathematical
set; (4) the response selection processes that compete to drive specificationofadynamicneuralfieldinthenextsection.
a Go or Nogo decision; (5) the motor planning processes that Themodelhasavisualfieldinthelowerrightpanelthatmimics
areactivated,eitherpartiallyorwhollybytheresponseselection propertiesofearlyvisualcorticalfields(Markounikauetal.,2008).
system; and (6) the motor control processes that do the job of Thevisualfieldiscomposedofneuralsitesreceptivetobothcolor
pushingtheresponsebuttonintheeventofaGodecision(whether (hue) and spatial position. Inputs into this field build localized
correctornot). ‘peaks’ofactivationinthetwo-dimensionalfieldthatspecifythe
IncognitivemodelingoftheGnGtask,modelstypicallyfocus colorofthestimulusandwhereitislocated.Thesepeaks,inturn,
ontheheartofthislist—theresponseselectionprocesses.Classic driveactivation–inparallel–inthefieldsalongaventralfeature
race-horsemodels(Boucher,Palmeri,Logan,&Schall,2007;Logan, pathway shown in the bottom row of Fig. 2 (see fAtn, con, wm)
Yamaguchi, Schall, & Palmeri, 2015), for instance, capture many andinadorsalpathwayinthetoprightpanel(seesAtn).Twoof
aspects of reaction time (RT) distributions from the GnG task thesefieldsare‘winner-take-all’attentionalfieldsthatselectively
usinganelegantsetofsimpleequations.Thesemodelshavealso attend to the color of the presented item (feature attention or
generated interesting neural predictions. More complex models fAtn)oritsspatialposition(spatialattentionorsAtn).Thesefields
havealsoconsideredaspectsofthememoryandvisualcomparison do not have much to do in the GnG task because only a single
processesthatunderlieperformanceinthistask(Wiecki&Frank, item is presented centrally in the visual field; they are included
2013). The Wiecki and Frank model, for instance, used a set of hereforcontinuitywithpreviousmodels(Schneegansetal.,2014;
SRassociationsinacomplexneuralnetworktoimplementthese Schöner et al., 2015) and to pass neurally-realistic inputs to the
memoryandvisualcomparisonprocesses.Thisaddedcomplexity othercorticalfields.
was justified because their goal was to mimic properties of the The more interesting fields are ‘higher up’ in the ventral
neuralsystemsthatunderlieresponseselection. pathway, where the model must decide whether the presented
Ourgoalinthepresentreportwastobuildaneuraldynamic colorisfromtheGosetortheNogoset.Thisrequiressomeform
model of response selection that captures the processes that of memory – the system has to remember the details of the Go
underlie the GnG task from perception to decision—to create an and Nogo set (see Logan et al., 2015 for evidence that the Nogo
integratedneuralarchitecturetocaptureprocesses1–4inthelist setisremembered)–andsomeformofvisualcomparison–the
above. (Links to motor planning and control systems have been systemhastovisuallycomparethehuevalueofthepresentedcolor
studiedextensivelywithDFT,butweoptedforsimplicityonthis tothememorizedoptions.Thereciprocallyinhibitoryarchitecture
front; for discussion, see Bicho & Schöner, 1997; Schöner et al., instantiatedintheworkingmemory(wm)andcontrast(con)fields
2015.)Wedidthisfortwocentralreasons.First,wehaveproposed implements this visual comparison process (see Johnson et al.,
and tested models that capture the full sweep of processes 1–4 2009). This piece of the architecture has been tested in several
inthedomainofVWM;thus,wewantedtoexaminewhetherthe previous studies including tests of novel behavioral predictions
processesthatunderlieperformanceinVWMtasksmightalsoplay (seeJohnsonetal.,2009).Moreover,thiscoreapproachtovisual
aroleinresponseselection.Thisisimportanttheoreticallybecause comparison has been generalized to visual comparison tasks
itprobesthegeneralityofatheory—canatheoryinstantiatedin in infancy as well (Perone & Spencer, 2013a,b, 2014). To this,
a particular architecture and designed to capture data from one we add a memory trace mechanism that remembers the colors
domain quantitatively capture data from a different domain of previouslyconsolidatedinworkingmemory(mem_wm)andthe
study? If so, this suggests that the model has the potential to colorspreviouslyidentifiedas‘contrasting’withthegosetinthe
integrate findings across domains provided, of course, that the contrast field (mem_con) (Lipinski, Schneegans, Sandamirskaya,
model is constrained and unable to capture findings that are Spencer, & Schöner, 2012; Perone, Simmering, & Spencer, 2011;
not presentinthosedomains.Notethatansweringthisquestion Schutte&Spencer,2002).
requiresdeepstudyofthetheoryinquestion.Wedonotdothat The final piece of the architecture implements the decision
workhere;rather,thepresentpaperismerelyafirststepinthis process. Here, we have implemented two dynamical nodes –
direction. localizedneuralpopulations(Schöneretal.,2015)–thatcompete
The second reason stems from Buss et al. (2013) where we in a winner-take-all manner to make a Go or a Nogo decision.
usedadynamicneuralfieldmodeltosimulatefMRIdatafroma The go node receives the summed activation from the working
dual-taskparadigm.Inthatproject,wediscoveredthatnon-neural memorylayer.Conceptually,iftheworkingmemorylayerdetects
inputs to the model – for instance, a perceptual input applied amatchbetweentherememberedsetofGocolors(inthememory

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 217
Fig.2. ArchitectureoftheGnGDNFmodel.Sevensub-networksareincluded:(i)thevisualfield,vis;(ii)thespatialattentionfield,sAtn;(iii)thefeatureattention,fAtn;
(iv)thecontrastfield,con;(v)theworkingmemoryfield;wm;(vi)thegoand(vii)nogonodes.Theneuralfieldsarecoupledbyuni-orbi-directionalexcitatory(green)or
|     |     |     |     |     |     | =t˜isplottedinblue.Fieldoutputg(u(x,t))att |     | =t˜isinred.The |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | -------------- | --- |
inhibitory(red)connections.Withineachfield,theactivationvariableu(x,t)atagiventimeinstancet
range[−20,20](horizontalaxisforfAtn,con,wm),or[−15,15],[−15,30](verticalaxisforsAtn,go,Nogo)showsvaluestakenbyactivationsandfieldoutputs.Feature
(color)andspacedimensionshaveaspanof204units(verticalaxesinthelowerpanels)and101units(horizontalaxesinupperandlowerrightpanels)respectively.(For
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
trace)andthecurrentcolordetectedinthefeatureattentionand and connectivity kernels c ( x ) with the latter defined either as a
visualfields,thislayerwillbuildapeakofactivation,consolidating GaussianfunctionorasthedifferenceoftwoGaussians(‘‘Mexican
theiteminworkingmemoryandpassingstrongactivationtothe hat’’shape).Thetemporaldynamicsoftheneuralactivityisalso
go node (Fig. 3(A)). Alternatively, if the contrast layer detects a influencedbyexternalinputss j anditisnon-deterministicdueto
η
| match between | the remembered | set | of Nogo colors | – the items | noise j . |     |     |     |     |
| ------------- | -------------- | --- | -------------- | ----------- | --------- | --- | --- | --- | --- |
|               |                |     |                |             |           | (   | , ) |     |     |
thatcontrastwiththeGoset–andthecurrentcolordetectedin The activation u x t is distributed continuously over an
thefeatureattentionandvisualfields,thislayerwillbuildapeak appropriatefeaturespacexsuchascolororspatialposition(Fig.2
( ( , ))
ofactivationandsendstrongactivationtothenogonode(Fig.3(B)). — blue curves). Then the field output g u x t , is computed
|     |     |     |     |     |     |     | (   | ) /( | Exp[−β u]) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---------- |
Conceptually,thewinnerintheracebetweenGoandnogonodes by the sigmoid (logistic) function g u = 1 1 +
wouldthendriveactivationinthemotorsystem(whichwedonot with threshold set to zero and steepness parameter β (Fig. 2—
implementhere). redcurves).Therefore,g ( u ) remainsnearzeroforlowactivations;
Inthesectionbelow,weprovideamoreformaltreatmentofthe it rises as activation reaches a soft threshold; and it saturates
|     |     |     |     |     | at a value | of one for high | activations. | Excitatory and | inhibitory |
| --- | --- | --- | --- | --- | ---------- | --------------- | ------------ | -------------- | ---------- |
dynamicneuralfieldmodel.Wealsowalkthroughanexampleto
illustratetheneuralpopulationdynamicsinthemodelthatgive coupling, both within fields and among them, promote the
rise to an in-the-moment decision to make a Go decision or to formationoflocalizedpeaksofactivationinresponsetoexternal
inhibitrespondingviaaNogodecision. stimulation. In our model, any above-the-threshold activation
|     |     |     |     |     | peak is interpreted | as  | an experimentally | detectable | (via neural |
| --- | --- | --- | --- | --- | ------------------- | --- | ----------------- | ---------- | ----------- |
recordings)responseofthatparticularneuralfieldtoastimulus.
4.2. Formalspecificationofthemodelandexemplarysimulations
Thearchitectureofthedynamicneuralfieldmodelincludesthe
|     |     |     |     |     | seven fields | shown in | Fig. 2. (For details | on field equations | and |
| --- | --- | --- | --- | --- | ------------ | -------- | -------------------- | ------------------ | --- |
Themodelconsistsofseveraldynamicneuralfields(DNFs)that
parametervalues,seeAppendix)Atimesnapshotofthedynamics
| computeneuralpopulationdynamicsu |     |     | accordingtothefollowing |     |                                                    |     |     |     |     |
| -------------------------------- | --- | --- | ----------------------- | --- | -------------------------------------------------- | --- | --- | --- | --- |
|                                  |     |     | j                       |     | oftheDNFmodelduringaGo/NogotaskisshowninFig.3.(The |     |     |     |     |
equation(Amari,1977;Ermentrout,1998):
|     |     |     |     |     | timeinstancetisapproximately500msafterstimulusonset,and | ˜   |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- |
τ u˙ ( x , t ) = −u ( x , t )+h +[c ∗g ( u )]( x , t ) itisindicatedonthegraphbyablackarrow.)
| e j | j   | j j j | j   |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
+  [c ∗g ( )]( , )+η ( , )+s ( ). Fig.3(A)illustratesthenetworkstateoftheDNFmodelattime
|     |     | u x t | x t | x (4.1) | ˜   |     |     |     |     |
| --- | --- | ----- | --- | ------- | --- | --- | --- | --- | --- |
jk k k j j t during the Gotask. The parameter values used in simulations
k
arelistedintheAppendix(Model1forLoad2condition).Shortly,
Theactivationu ofeachcomponentismodeledathightemporal whenaGocolorispresented(durationofstimulusis1500ms),an
|     | j   |     |     | τ   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
resolution (millisecond timescale) with time constant . It activationpeakisbuiltinthevisualfieldvis.Thisinducesapeak
e
assumes a resting level h and depends on lateral (within the intheworkingmemoryfieldwmandaweakpeakinthefeature
j
field) and longer range (between different fields) excitatory and attentionfieldfAtn(curvesinblue).Then,thepeakinwmleads
inhibitoryinteractions,c∗g ( u ) andc ∗g ( u ) respectively.These to an increase in activation of the go node (Fig. 3(A); in green).
|     |     | j j j | jk k k |     |     |     |     |     |     |
| --- | --- | ----- | ------ | --- | --- | --- | --- | --- | --- |
areimplementedbyconvolutionsbetweenfieldoutputsg ( u ( x , t )) Inaddition,duetoinhibitionfromwmthatdominatesexcitation

218 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
Fig.3. NetworkstateoftheDNFmodelattimeinstancet˜approximately500msafterstimulusonsetduring:(A)Gotaskand(B)Nogotask(onlyvis,fAtn,con,wmare
shown).Timeevolutionoftheoutputofgo(ingreen)andNogo(inred;leftpanel)nodesisalsoshown.Timet˜isindicatedbytheblackarrow.Simulationsusedparameters
fromAppendix(seeModel1andLoad2condition).(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
receivedfromvis,theactivityofthecontrastfieldconislowered in a manner that is consistent with our hypotheses about how
at the location of the Go color. At some time between 400 and response selection works. When contrasted with biophysical
500 ms after stimulus onset, the activity of the go node crosses neuralnetworkmodels,however,thedynamicneuralfieldmodel
thethreshold,thatis,itsoutputfunctionisgreaterthan0.5(see is relatively simple—there are fewer neural sites and far fewer
leftpanel;ingreen).Thisiscausedbytheformationofastrong free parameters. Along this dimension of complexity, therefore,
peakinwm.Inaddition,thepeakinfAtnbecomesstrongerand DFTsitssomewhereinthemiddle.Thatisbydesign.Wecontend
a sub-threshold hill forms in con as well. In the interval of time that using neural process models is critical in psychology and
betweentheresponse(reactiontimeRT∼450ms)andendofthe neurosciencebecausethisopensthedoortoimportantconstraints
trial(1500ms),theactivitypeaksinvis,fAtn,conandwmstabilize. fortheoryfrombothbehavioralandneuralmeasures—constraints
Importantly,thehillinconremainssub-threshold.Alsonotethat readilyapparentwhenonetriestoconstructintegrativecognitive
theactivityofthegonodereachessaturation. neuroscience models. In our view, these constraints justify the
˜ complexity. At the same time, we think it is important to add
Fig.3(B)showsthenetworkstateoftheDNFmodelattimet
justtherightamountofcomplexity.Datafromneurophysiology
duringtheNogotask.Inthiscase,theNogocolorinducesactivation
suggesttousthatperception,cognition,andactionplanningliveat
of the visual field vis. This, in turn, increases activation in the
thelevelofneuralpopulationdynamics,andnotatthebiophysical
contrastfieldconatthecorrespondingcolorcoordinatealongthe
level per se (for discussion, see Schöner et al., 2015). Thus, we
featurespace.Asub-thresholdhillinfAtnformsaswell,andwm
˜ contend that the added detail from biophysical models is not
islocallyinhibited.Then,laterduringthetrial(e.g.attimet),the
criticalifthegoalistobridgethegapbetweenbrainandbehavior.
activationofthenogonodehascrosseditsthreshold.Thepeakin
Ofcourse,thedownsidetotheaddedcomplexityintroducedby
conbecomesstrongerandstabilizes,andfieldfAtnshowssupra-
dynamicneuralfieldmodelsisthatfittingdatatobehavioraland
thresholdactivity.AttheNogocolorlocationinwmtheactivityis
neuraldatabecomesharderandabitmoresubjectiveinnature.
inhibited.Approachingtheendofthetrial,theactivitystabilizesin
ThisisnottosaythatDFTcannotachievequantitativefits—that
vis,fAtn,conandwm,thepeakinwmremainingsub-threshold.
iscertainlystillagoal.Rather,thesubjectivesenseofDFTcomes
Note that the nogo node stays ‘on’, while the go node remains
fromthefactthatitisrarelypossibletosearchthefullparameter
inactive.
spaceofadynamicneuralfieldmodel.Consequently,manyofthe
issuesthatarecentraltomathematicalpsychologyandmanyofthe
5. Simulatingbehaviorwiththedynamicneuralfieldmodel toolsthatareusedtoevaluatemodelfits(Turneretal.,2016)are
difficult,ifnotimpossible,toapplytodynamicneuralfieldmodels
When contrasted with cognitive models, the dynamic neural (Samuelsonetal.,2015).
fieldmodelinFig.2iscomplex.Eachfieldhasseveralparameters Critically,however,fittingdynamicneuralfieldmodelstodata
thatneedtobe‘tuned’appropriatelytogetthemodeltoperform isnotanunconstrainedfree-for-all.Rather,constraintscomefrom

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 219
multiple sources. First, the neural dynamics in the model must 5.1. Simulationmethods
| reflect our | understanding |     | of how brains | work. | Thus, | we would |     |     |     |     |     |     |     |     |
| ----------- | ------------- | --- | ------------- | ----- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Beforeturningtothedetailsofthebehavioralfits,weprovide
ruleoutparametersthatgiverisetopathologicalneuralstates.For
instance, if excitatory neural interaction strengths in one of the a few more details about the simulation method. All numerical
cortical fields are too strong, input to the field will build a peak simulations were performed using the COSIVINA simulation
thatgrowsoutofcontrol—themodelhasaseizure.Bycontrast,if package(availableatwww.dynamicfieldtheory.org).Thispackage
excitatoryneuralinteractionstrengthsaretooweak,nopeakswill allows one to construct dynamic neural field architectures
build—themodelwillremaininasub-thresholdstate. relativelyquickly,alongwithagraphicuserinterfacethatenables
Second, parameters must be tuned such that the neural evaluationand‘tuning’ofthemodelinrealtime(seeFigs.2–3).
dynamicsreflectourconceptualtheoryofhowthemodelshould The same simulator can then be run in ‘batch’ mode to iterate
|     |     |     |     |     |     |     | the model | across | many | trials, recording |     | responses | that | can be |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ---- | ----------------- | --- | --------- | ---- | ------ |
behaveinthetask.Concretely,thismeansthattherightsequence
|          |         |            |        |            |         |             | evaluated | relative | to empirical | data. | The | COSIVINA | package | also |
| -------- | ------- | ---------- | ------ | ---------- | ------- | ----------- | --------- | -------- | ------------ | ----- | --- | -------- | ------- | ---- |
| of peaks | emerges | during the | course | of a trial | to give | rise to the |           |          |              |       |     |          |         |      |
includesanewtoolboxforgeneratinglocalfieldpotentialsdirectly
righttypeofbehavior(inthiscase,thegenerationofaGoorNogo
fromthemodelatthesametimethatthemodelissimulatingthe
decision).Formally,thismeansthatthesequenceofbifurcations
experimentaltask.Thus,themodelistrulyanintegrativecognitive
inthemodelmustbecorrect.Forinstance,thefollowingshould
hold:(1)peaksintheworkingmemoryandcontrastfieldsshould neurosciencemodel,generatingbehavioralandneuraldata(with
not build spontaneously from a memory trace; (2) peaks in the millisecondprecision)simultaneously.
| working | memory | and contrast | fields | should | be influenced | by  |     |     |     |     |     |     |     |     |
| ------- | ------ | ------------ | ------ | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
5.1.1. ParameterfittinginModel1
| the formation | of  | peaks in | feature attention |     | (that is, the | parallel |     |         |               |     |          |      |        |       |
| ------------- | --- | -------- | ----------------- | --- | ------------- | -------- | --- | ------- | ------------- | --- | -------- | ---- | ------ | ----- |
|               |     |          |                   |     |               |          | We  | adopted | the following |     | approach | when | tuning | model |
inputfromthevisualfieldshouldnotbetoostrong);and(3)the
GoandNogocompetitionshouldbeinfluencedbysub-threshold parameterstoarriveatModel1.First,wemadeasimplificationof
activationintheworkingmemoryandcontrastfieldsasdecision- themodel.Initialsimulationswithadynamicmemorytraceinboth
makingunfolds. theworkingmemoryandcontrastfieldsshowedthatthememory
Thethirdcategoryofconstraintcomes,ofcourse,fromthede- tracedynamicsconformedtoexpectationsbasedonpreviouswork
tailsofbehavioraldata.IntheGnGtask,theseconstraintsarerel- (Bussetal.,2013;Erlhagen&Schöner,2002;Lipinskietal.,2010).
atively modest since the participant only responds on Go trials. Inparticular,memorytraceswerestrongerintheLoad2condition
|               |     |               |                  |     |        |           | and weaker | in        | the Load | 6 condition. |        | This occurs | because | each       |
| ------------- | --- | ------------- | ---------------- | --- | ------ | --------- | ---------- | --------- | -------- | ------------ | ------ | ----------- | ------- | ---------- |
| Nevertheless, | if  | one considers | RT distributions |     | rather | than just |            |           |          |              |        |             |         |            |
|               |     |               |                  |     |        |           | color is   | presented | more     | often over   | trials | in          | Load 2. | Similarly, |
means,thiscanberelativelyconstraining.Forinstance,Erlhagen
memorytraceswerestrongerforGostimuliintheProportion75%
andSchönerfitthedetailsofresponsedistributionsfromseveral
|     |     |     |     |     |     |     | condition | and | weaker | in the 25% | condition. |     | Again, this | mimics |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------ | ---------- | ---------- | --- | ----------- | ------ |
responseselectionparadigms(Erlhagen&Schöner,2002)Thisis
|     |     |     |     |     |     |     | the frequency |     | of stimulus | presentation. |     | Although | these | memory |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | ------------- | --- | -------- | ----- | ------ |
possiblewithdynamicneuralfieldmodelsbecausesuchmodels
arestochastic,andtheygeneratemeasurablebehaviorsonevery trace – or learning – dynamics are fundamentally interesting,
trial(e.g.,theformationofastableGoorNogodecision).Moreover, theyalsomakesimulationworkmorecomplexbecauseonemust
relativelycomplexmodelsastheoneusedheregeneratecomplex simulateavarietyofstimuluspresentationorderstoobtainrobust
estimatesoflearningeffects.Giventhatsuchlearningeffects–in
non-linearpatternsthroughtime—forinstance,asequenceofpeak
bothbehavioralandfMRIdata–werecentraltoourpreviouswork
statesacrossfields,whichcanamplifystochasticfluctuationslead-
usinganinteractivemodel-basedfMRIapproach(Bussetal.,2013),
ingtomacroscopicbehavioraldifferencesacrossconditions.Fur-
weoptedtosimplifythelearningdynamicshere.Thus,insteadof
therbehavioralconstraintsemergewhenoneconsidersresponse
simulatingmemorytracesdynamicallyovertrials,weusedstatic
distributionsfrommultiplestudies.Here,thegoalwouldbetocap-
|     |     |     |     |     |     |     | memory | traces, | that is, | the memory | trace | inputs | were | fixed for |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------- | -------- | ---------- | ----- | ------ | ---- | --------- |
turethequantitativedetailsofbehavioralresponsesfrommulti-
|     |     |     |     |     |     |     | each condition |     | to reflect | the properties |     | revealed | by these | initial |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ---------- | -------------- | --- | -------- | -------- | ------- |
plestudies,ideallywithoutanymodificationtomodelparameters.
|     |     |     |     |     |     |     | simulations | (see | Eq. (A.17) | and | Table | A.4.1in | the Appendix, | for |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ---------- | --- | ----- | ------- | ------------- | --- |
Thishasbeenachievedinseveralnotablecases(Buss&Spencer,
details).
2014;Erlhagen&Schöner,2002;Schutte&Spencer,2002).
|       |           |           |           |     |                  |     | The | next objective |     | was to | find a | set of | parameters | that |
| ----- | --------- | --------- | --------- | --- | ---------------- | --- | --- | -------------- | --- | ------ | ------ | ------ | ---------- | ---- |
| Here, | our goals | were more | modest—we |     | did not optimize | the |     |                |     |        |        |        |            |      |
quantitativelycaptureddatafromtheLoad2conditionWestarted
quantitativefittothebehavioraldata.Rather,wepursuedamore
|     |     |     |     |     |     |     | with parameters |     | from | Schöner | et al. | (2015, | Chapter | 8), and |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ---- | ------- | ------ | ------ | ------- | ------- |
iterativeparameterfittingapproach.First,wefitthemeanreaction
adjustedthemodelparameterstoapproximatetherightbehavior
| times with | the dynamic | neural | field | model, | and made | sure the |          |      |              |     |           |            |     |           |
| ---------- | ----------- | ------ | ----- | ------ | -------- | -------- | -------- | ---- | ------------ | --- | --------- | ---------- | --- | --------- |
|            |             |        |       |        |          |          | from the | Load | 2 condition. | For | instance, | connection |     | strengths |
varianceinthemodelwasintherightballpark.Werefertothisas
betweenthegonodeandwmfieldandnogonodeandconfieldwere
Model1(seeAppendix).Asreaderswillsee,ourfitstothestandard
tuned.Thestrengthofthememorytraceinputsintothewmand
deviationscouldhavebeenbetter;however,wedidnotoptimize
confieldsforGoandNogotrialsrespectively,weretunedaswell.
the model on this front. Rather, we pushed forward to evaluate OncethemodelcapturedthereactiontimesforGotrialsatLoad
thequantitativefMRIfitsfirst.Datafromthesefitsrevealedthat 2,thenextstepwastocapturereactiontimesfortheLoad4and
Model1didnotquiteoutperformthequantitativefitprovidedbya Load6conditionsHerewehypothesizedthatincreasingtheLoad
StandardGLManalysis—the‘goldstandard’statisticalmodelwe in the task would increase competition among memory traces,
setapriori.Wethenexaminedthemodel’sneuraldata,focusing
|     |     |     |     |     |     |     | slowing | down | the time | it takes | to build | a peak | in the | working |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---- | -------- | -------- | -------- | ------ | ------ | ------- |
onthewaysinwhichthemodel’sneuraldynamicsdifferedfrom
|            |          |         |        |           |                 |     | memory | and contrast |     | fields and | yielding | slower | reaction | times |
| ---------- | -------- | ------- | ------ | --------- | --------------- | --- | ------ | ------------ | --- | ---------- | -------- | ------ | -------- | ----- |
| the neural | dynamics | evident | in the | fMRI data | (see Wijeakumar |     |        |              |     |            |          |        |          |       |
(Erlhagen&Schöner,2002).Hence,weadjustedthestrengthofthe
etal.,2015).Thisledtonewinsightsintohowwehadthemodel
memorytraceinputsinbothwmandconfieldswithoutmodifying
parameters‘tuned’andpromptedasecondroundofbehavioralfits any other parameters. (See Table A.4.1 in the Appendix; third
targetingmorecompetitiveneuralinteractions.Thisresultedina column showshow the strengthof the memorytrace inputs for
secondsetofparameters–Model2(seeAppendix)–thatfitthe wmandconisvariedacrossdifferentconditions.)Wethentested
behavioral data relatively well and fit the fMRI data better than whether the model was able to capture the increase in reaction
Model1.Thisillustrateshowaninteractivecognitiveneuroscience timesobservedasmemoryLoadincreased.
approachcanbeusedinpracticetobridgethegapbetweenbrain FortheProportionmanipulation,Proportion50%corresponded
andbehavior. to Load 4 and so its parameters were used as an anchor to fit

220 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
|     | 650 |     |     |     |     |     | 650 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | A   |     |     |     |     |     | B   |     |     |     |     |     |     |
|     | 600 |     |     |     |     |     | 600 |     |     |     |     |     |     |     |
|     | 550 |     |     |     |     |     | 550 |     |     |     |     |     |     |     |
|     | 500 |     |     |     |     |     | 500 |     |     |     |     |     |     |     |
|     | 450 |     |     |     |     |     | 450 |     |     |     |     |     |     |     |
)sm( semit noitcaeR
|     | 400 |        |        |     |        |     | 400 |         |     |         |     |         |     |     |
| --- | --- | ------ | ------ | --- | ------ | --- | --- | ------- | --- | ------- | --- | ------- | --- | --- |
|     |     | Load 2 | Load 4 |     | Load 6 |     |     | Prop 25 |     | Prop 50 |     | Prop 75 |     |     |
|     | 180 |        |        |     |        |     | 180 |         |     |         |     |         |     |     |
|     |     | C      |        |     |        |     |     | D       |     |         |     |         |     |     |
|     | 120 |        |        |     |        |     | 120 |         |     |         |     |         |     |     |
|     | 60  |        |        |     |        |     | 60  |         |     |         |     |         |     |     |
|     | 0   |        |        |     |        |     | 0   |         |     |         |     |         |     |     |
|     |     | Load 2 | Load 4 |     | Load 6 |     |     | Prop 25 |     | Prop 50 |     | Prop 75 |     |     |
Fig.4. (A–B)MeanreactiontimescomputedfortheDNFmodel(Model1showninlightgrayandModel2shownindarkgray)andbehavioraldata(showninblack)for
themanipulationofthe(A)Loadand(B)Proportion.(C–D)Meanstandarddeviationsofreactiontimesacrosssimulationsforthe(Model1showninlightgrayandModel2
shownindarkgray)andbehavioraldata(showninblack)forthemanipulationof(C)Loadand(D)Proportion.
thereactiontimesfromProportion25%andProportion75%.Here, nodeandconfieldsotheywereequaltotheparametersconnecting
we hypothesized that as the number of Go trials increased, the gonodeandwmfield(TableA.3.1).
strength of the memory trace for Go trials would also increase. Ourexaminationofthemodel’sneuraldynamicsalsorevealed
Likewise, as the number of Go trials decreased, the strength of that differences across conditions were relatively modest. We
thesememorytraceswoulddecrease(TableA.4.1intheAppendix). realized that this was influenced by the trial duration we were
To generate quantitative data from the model, we ran 144 simulating.Decisionsinthemodel–anddecisionsbyparticipants
trials per model and 20 identical models (to reflect the number – occur within the first 500 ms; for the remaining 1000 ms,
of participants in the original study) for each of the Load and the model simply sits in a neural attractor state, maintaining
Proportion manipulations. Mean and standard deviations were peaksacrossallfields(becausethestimulusremains‘on’).Because
| calculated | across | reaction times | and compared |     | to the empirical |     |          |        |          |     |            |      |             |        |
| ---------- | ------ | -------------- | ------------ | --- | ---------------- | --- | -------- | ------ | -------- | --- | ---------- | ---- | ----------- | ------ |
|            |        |                |              |     |                  |     | the BOLD | signal | reflects | the | slow blood | flow | response to | all of |
data(Fig.4). these events, the ‘final’ attractor states of the model dominate
thehemodynamicpredictionsandthemoreinterestingcognitive
5.1.2. ParameterfittinginModel2 processes – the neural interactions leading to the decision –
ToidentifyparametersforModel2,weproceededasfollows. haverelativelylessimpact.Thisdoesnotaccuratelyreflectneural
|                   |          |             |           |          |                  |        | systems;  | rather, | neurophysiological |     |          | data suggest | that   | neural |
| ----------------- | -------- | ----------- | --------- | -------- | ---------------- | ------ | --------- | ------- | ------------------ | --- | -------- | ------------ | ------ | ------ |
| After discovering |          | that Model  | 1 did not | meet     | our quantitative |        |           |         |                    |     |          |              |        |        |
|                   |          |             |           |          |                  |        | attractor | states  | stabilize,         | but | are then | suppressed   | once a | stable |
| criterion         | for fits | to the fMRI | data, we  | examined | the              | neural |           |         |                    |     |          |              |        |        |
decisionhasbeenmade(Bastianetal.,2003).Toimplementthis,
| predictions  | from       | the model | across conditions |           | relative to | fMRI |          |     |            |                  |     |             |          |     |
| ------------ | ---------- | --------- | ----------------- | --------- | ----------- | ---- | -------- | --- | ---------- | ---------------- | --- | ----------- | -------- | --- |
|              |            |           |                   |           |             |      | we added | a   | ‘condition | of satisfaction’ |     | node (CoS), | building | off |
| results from | Wijeakumar |           | et al. (2015).    | A central | effect      | in   |          |     |            |                  |     |             |          |     |
recentworkbySandamirskayaandcolleagues(Sandamirskaya&
Wijeakumaretal.wasthatregionsofthefronto-cortical–striatal
|     |     |     |     |     |     |     | Schöner, | 2008; | Sandamirskaya, |     | Zibner, | Schneegans, | & Schöner, |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | -------------- | --- | ------- | ----------- | ---------- | --- |
networkshowedgreateractivationoninfrequenttrials,regardless
ofwhetheraninfrequentstimulusappearedonaGoorNogotrial 2013; Schöner et al., 2015). This node receives input from both
(Wijeakumar et al., 2015). For instance, brain areas responded the go and nogo nodes. When either becomes active, the ‘CoS’
strongly on infrequent Go trials. Quantitative fMRI predictions node becomes active, signaling that the conditions for a stable
from Model 1 did not show this pattern. Given that local decision have been satisfied. The CoS node then suppresses the
field potentials are positively influenced by both excitatory and working memory and contrast fields, globally inhibiting these
inhibitory interactions, we hypothesized that a strong response fields.Consequently,thestabledecisionmadebythegoornogo
|               |     |              |         |        |          |      | node | remains | active | throughout | the | 1500 ms | trial, but peaks | in  |
| ------------- | --- | ------------ | ------- | ------ | -------- | ---- | ---- | ------- | ------ | ---------- | --- | ------- | ---------------- | --- |
| on infrequent | Go  | trials might | be most | likely | to occur | when |      |         |        |            |     |         |                  |     |
thewmandconfieldsaresuppressedoncethedecisionismade.
thereisastrongmemoryoffrequentNogoresponsesandstrong
|     |     |     |     |     |     |     | Conceptually, |     | this frees | up  | these systems | to  | move on to | other |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------- | --- | ------------- | --- | ---------- | ----- |
competitionbetweentheworkingmemoryandcontrastfields(and
interestingeventsthatmight(butdonot)occurinthevisualfield.
viceversaoninfrequentNogotrials).Toexaminethispossibility,
| we added | a new | element to | the model—a | memory | trace | to the |     |     |     |     |     |     |     |     |
| -------- | ----- | ---------- | ----------- | ------ | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
go and nogo nodes (implemented by modulating the gain on 5.2. Quantitativebehavioralresults
self-excitationacrossconditions,seeTableA.2.1intheAppendix)
and we increased competition between the wm and con fields Here,wepresenttheresultsofthebehavioralfitsforModels1
(TableA.3.1).Wealsobalancedtheparametersacrossthegoand and2alongsidethereactiontimesfromtheactualbehavioraldata
Nogo systems, setting the reciprocal connections between nogo BothDNFmodelsprovidereasonablefitstothetrendsinreaction

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 221

timesshownbythebehavioraldatainresponsetomanipulating + 1  ∗g  ( , )
|     |     |     |     |     |     |     |     |     | c con,fAtn |     | u   | y t  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | ----- | --- | --- |
ProportionandLoad(seeFigs.4(A)and(B)).RootMeanSquared n fAtn fAtn
E rr or (R M S E ) fo r r e ac t i o n t im es fo r M o d e l 1 w i th r es p e c t t o  (uwm )( , )
|            |             |       |            |           |            |        |                 |     | + c con,wm ∗gwm |     | y t |    |     |     |
| ---------- | ----------- | ----- | ---------- | --------- | ---------- | ------ | --------------- | --- | ---------------- | --- | --- | --- | --- | --- |
| th e Sta n | da r d G LM | a n a | l y s is = | 1 0.5 8 m | s a n d RM | S E fo | r r e a c tio n |     |                  |     |     |     |     |     |

|     |     |     |     |     |     |     | =   |     | 1   |    | (uvis | )( , ) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------- | --- | --- |
t i m e s f or M o d e l 2 w it h r es p e c t t o t h e S t a n d a r d G LM an a l y s is + c con,vis ∗gvis y t dy
n× m
| 2 7 .0 2 m | s. F o r t | h e Lo a | d m a n i p | u l ati o n , | r e a c ti o n | tim es | in c r e a se d |     |     |     |       |     |     |     |
| ---------- | ---------- | -------- | ----------- | ------------- | -------------- | ------ | --------------- | --- | --- | --- | ----- | --- | --- | --- |
|            |            |          |             |               |                |        |                 |     |    |     | (t) |     |     |     |
as the number of SR mappings increased. For the Proportion +  .
|     |     |     |     |     |     |     |     |     | a con,nogo g | nogo u nogo |    |     |     | (6.1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ----------- | --- | --- | --- | ----- |
manipulation,increasingthefrequencyofGotrialsfrom25%to75%
resultedinadecreaseinreactiontimes.Althoughthereweresome Several observations about this calculation need to be made.
variationsinthestandarddeviationsacrossthe20simulationsfor First,sincebothexcitatoryandinhibitorycommunicationrequire
bothmodels(asshowninFig.4(C)and(D)),thetrendsacrossthe activeneuronsand,biophysically,generatepositiveionflow,we
conditionswerequalitativelycorrect. need to sum both in a positive way toward predictions of local
|     |     |     |     |     |     |     |     | activity; thus, | we take | the absolute | value | of all | excitatory | and |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ------- | ------------ | ----- | ------ | ---------- | --- |
inhibitorycontributions.Second,giventhatfieldactivitiesinthe
| 6. Generating | local | field | potentials | and | hemodynamics |     | from |             |            |         |          |           |             |     |
| ------------- | ----- | ----- | ---------- | --- | ------------ | --- | ---- | ----------- | ---------- | ------- | -------- | --------- | ----------- | --- |
|               |       |       |            |     |              |     |      | calculation | of the LFP | measure | may span | different | dimensions, |     |
theDNFmodel
|     |     |     |     |     |     |     |     | we normalize | them. In | this way, | we  | can maintain | a   | balance |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --------- | --- | ------------ | --- | ------- |
Tosimulatethehemodynamicsforthisstudy,weadaptedthe among their contributions. We do that by dividing each field
model-basedfMRIapproachfromDecoetal.(2004).Specifically, contribution by the number of units in it (e.g., in Eq. 6.1 certain
×
we created an LFP measure for each component of the model field contributions were divided by n or n m where n is the
during each condition and tracked the LFPs in real time as feature dimension and m is the space dimension). Third, due to
the model simulated behavioral data. Then, we convolved the correlatednoiseineachfieldofthemodel,small-scalevariationsin
simulated LFPs with a gamma impulse response function to thesignaloccur(especiallyevidentinthesecondcomponent),as
generatesimulatedhemodynamics,andasaresult,regressorsfor wellasoverallvariationinreactiontimes.Indeed,forsameinitial
eachcomponentandcondition. conditions,theDNFmodelyieldsrelativelydifferentLFPmeasures
(seeFig.5(A)).
6.1. DefinitionoftheDNFmodel-basedLFP Each component in the model has a different network of
interactionsthatdrivesadifferentresponsepattern.Consequently,
Toillustratetheprocedure,weexplainbelowthecomputation individual LFP measures are created for each model component,
of the LFP for the contrast field neural population (con field in thatis,foreachofthe7fieldsshowninFig.2.Figs.5(A)and(B)
Figs.2–3).TheLFPsforallotherneuralfieldsintheGnGDNFmodel depictLFPsimulationsfromfAtnandgonodeinModel2,overthree
(e.g.Model1;seeFig.1)followanidenticalapproach. andfourtrials,respectively.
| Consider | the | dynamic | field | Eq. (4.1) | with appropriate |     | input |     |     |     |     |     |     |     |
| -------- | --- | ------- | ----- | --------- | ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
neuralfieldsandconnectionsthatcontributetothedynamicsof
|     |     |     |     |     |     |     |     | 6.2. CanonicalpredictedLFPsperexperimentalcondition |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
theneuralpopulationintheconfield.Thisequationisdefinedby
(A.4)intheAppendixor,moreexplicitly,by Notethat,insomecomponents,theLFPlevelissimilaracross
τ u˙ ( , ) = −u ( , )+h +s ( )+c ∗ξ( ) conditionswithminordifferencesintiming(fAtn).Inothers(go
| e con y | t   | con y | t con | con | y con,noise |     | yt  |     |     |     |     |     |     |     |
| ------- | --- | ----- | ----- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
+(( c −c )∗g ( u ))( y , t ) n o d e ) , d if f er e n t c o n d it i o n s (G o t r i a lv e r su s N o go t ri a l) l ea d to la r g e r
|     |     | co n,E | con,I | con | con |     |     |     |     |     |     |     |     |     |
| --- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 di f fe r e n c e s i n t h e L F P ( F i g .5 (B ) ) . T h is c o n tr a st is k e y t o th e m o d e l-
|     | +   |     |     | ∗g ( )( | , ) |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
c con,j j u j y t based approach because it allows components to have unique
j=vis,fAtn,wm signaturesonboththescaleoftheindividualtrialaswellaslarger
+a con,nogo g ( u ( t )) scalesignaturesacrosstaskconditions.
nogo nogo
|         |             |     |             |     |             |    |              | Toaccountforthisvariance,werunmanyrepetitionsofeach |     |     |     |     |     |     |
| ------- | ----------- | --- | ----------- | --- | ----------- | --- | ------------ | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| where f | ∗ h denotes | the | convolution | f   | ∗ h ( y , t | ) = | f ( y − y ′) |                                                     |     |     |     |     |     |     |
condition(i.e.westartfromsameinitialvaluesinthemodel;there-
h ( y ′, t ) dy. ′
fore,thevariabilitywillbeadirectconsequenceofnoiseonly).The
| Heres | ( y ) specifiesthestationarysub-thresholdstimulusto |     |     |     |     |     |     |                                                        |     |     |     |     |     |     |
| ----- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
|       | con                                                 |     |     |     |     |     |     | numberofrepetitionsischosenusuallytoreflectthenumberof |     |     |     |     |     |     |
theconfield(‘‘thememorytrace’’),spatiallytunedtoNogocolors.
|               |            |             |            | η      |            |                |       | trialsundertakenbythesubjectsintheactualexperiment.(Forex- |                 |        |         |                |     |          |
| ------------- | ---------- | ----------- | ---------- | ------ | ---------- | -------------- | ----- | ---------------------------------------------------------- | --------------- | ------ | ------- | -------------- | --- | -------- |
| The spatially | correlated |             | noise      | con is | obtained   | by convolution |       |                                                            |                 |        |         |                |     |          |
|               |            |             |            |        | ξ          |                |       | ample,ifintheexperiment,eachof20subjectsunderwent72Go      |                 |        |         |                |     |          |
| between       | kernel     | c con,noise | and        | vector | of white   | noise.         | Local |                                                            |                 |        |         |                |     |          |
|               |            |             |            |        |            |                |       | trials for                                                 | Load 4, we will | run 20 | sets of | 72 repetitions |     | (simula- |
| connections   | include    | both        | excitatory | and    | inhibitory | components,    |       |                                                            |                 |        |         |                |     |          |
= − t i o n s ) o f M o d e l 2 w i th t h e c o r r e s p o n d i n g p a r a m e t e r s f o r s t i m u lu s
| c c | on, c | ,I . A ll | k e r n e ls | ar e G a u s | s i a n f u n | c t io n s o | ft h e f o r m |     |     |     |     |     |     |     |
| --- | ----- | --------- | ------------ | ------------ | ------------- | ------------ | -------------- | --- | --- | --- | --- | --- | --- | --- |
c o n c E c on   s t r e n g t h f r o m T a b l e A .4 .1 . ) W e t h e n a v e r a g e t h e g e n e r a t e d L F P t im e
| ( − | ′ ) = | −   | ( y − y ′ )2 |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
c y y a E x p w i th p o s i t i v e p a r a m e t e rs a e x c e p t s e r ie s o v e r r e p e t i t i o n s o f t h e s a m e c o n d i t i o n t o d e t e r m i n e w h a t w e
2 σ 2
a < 0.Notethat,wheneverModel2isusedinsimulations, callthecanonicalpredictedLFPsignalpercondition.Fig.6depicts
con,wm
examplesofsuchcanonicalLFPpredictionsfortwofields,fAtn(in
anadditionaltermassociatedwithfeedbackprojectionsfromthe
blue)andgo-node(ingreen).Thefirst1500msinFig.6showsthe
| conditionofsatisfactionnode(CoS)appearsinu |     |     |     |     |     | con . |     |     |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
TogenerateanLFPforthecontrastfield,wesumtheabsolute canonicalLFPpredictionsforLoad4,Gotrials(e.g.,asseenrepeated
valueofalltermscontributingtotherateofchangeofactivation inFig.5(A)).Thelast1500msshowsthecanonicalLFPpredictions
|     |     |     |     |     | −u  | (   | )   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
within the field, excluding the stability term, yt , and the forLoad4Nogotrials.
con
| neuronalrestinglevel,h |     |     | .TheresultingLFPequationforthecon |     |     |     |     |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
con
fieldisgivenby:
|     |     |     |     |     |     |     |     | 6.3. Constructionofthelong-formLFPtemplate |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |

|         | 1   |        |                 |     | )        |     |     |                                                     |     |     |     |     |     |     |
| ------- | --- | ------ | ---------------- | --- | --------- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| LFP ( t | ) = | |s ( y | )|+ c con,noise | ∗ξ( | y , t dy |     |     |                                                     |     |     |     |     |     |     |
| con     |     | con    |                  |     |           |     |     | Anotherconcernthatweaimedtoaddresswasplacingthesim- |     |     |     |     |     |     |
n
|     |     |    |     |     |     |     |     | ulatedcanonicalLFPvaluesinanappropriatecontext.Muchlike |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     | 1   |    |     |     | )  |     |     |                                                         |     |     |     |     |     |     |
+ c ∗g ( u )( y , t  themeasurementoffMRIdata,wetakeabaselinemeasurement
|     |     | con,E | con | con |     |     |     |                                                        |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
|     | n   |       |     |     |     |     |     | fromthemodelasfollows.WeusethesameLFPcalculationsasde- |     |     |     |     |     |     |
|     |    |       |     | )  |     |     |     |                                                        |     |     |     |     |     |     |
+ c ∗g ( u )( y , t dy scribedabove,butwecomputea‘‘restinglevel’’bysimulatingthe
|     |     | con,I | con con |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

222 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
A 10 long-form,averagedLFPtemplates.Thelatterarelong-scale(tens
ofminutes)model-generatedLFPpredictionsforeachsubjectin
9
theexperiment.Thestructureofthelong-formLFPtemplates,for
8
|     |     |     |     |     |     |     | all components |     | of the | DNF | model, is | determined |     | by the | order |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------ | --- | --------- | ---------- | --- | ------ | ----- |
7
andtimingoftrialsthatparticularsubjectexperiencedduringthe
ytivitcA dleiF 6
experimentalblock(s).
|     | 5   |     |     |     |     |     | Todothis,wefirstcreateazero-valuedtimeseriesthelengthof |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
theentireexperiment(i.e.azero-valuedlong-formLFPtemplate).
4
|     |     |     |     |     |     |     | We then | use trial | onset | timings | from | the experiment |     | to  | anchor |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | ----- | ------- | ---- | -------------- | --- | --- | ------ |
3
thetrialcanonicalbaselinedLFPprediction,foreachcorresponding
|     | 2   |     |     |     |     |     | trialtype.Forexample,ifatrialofacertaincondition(e.g.Load |      |     |           |     |          |       |             |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------------- | ---- | --- | --------- | --- | -------- | ----- | ----------- | --- |
|     | 1   |     |     |     |     |     | 4,Nogotrial)hasanonsettimeof7500msafterthestartofthe      |      |     |           |     |          |       |             |     |
|     |     |     |     |     |     |     | experiment,                                               | then | the | canonical | LFP | for that | trial | is inserted | to  |
0
|     | 0   | 500 | 1000 1500 2000 | 2500 3000 | 3500 | 4000 4500 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | -------------- | --------- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thelong-formtemplate-LFPstartingatthesameonsettime(see
Time (milliseconds)
Fig.7).Oncethisiterativeprocessiscompleted(acrossalltrials)
B 10
andthealgorithmisappliedtoallDNFmodelcomponents,wehave
9 constructedexperiment-based,subject-specificLFPtimeseriesfor
|     | 8   |     |     |     |     |     | eachcomponentoftheDNFarchitecture.Thesetimeseriesreflect |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
predicteddifferencesinneuralactivationbasedontheprocesses
7
atworkwithineachfield.
ytivitcA dleiF 6
|     | 5   |     |     |     |     |     | 6.4. GeneratinghemodynamicsfromtheDNFmodel |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
4
|     | 3   |     |     |     |     |     | fMRIdatadoesnotmeasureneuralactivitydirectly.Itmeasures |          |      |        |               |     |        |          |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | -------- | ---- | ------ | ------------- | --- | ------ | -------- | --- |
|     |     |     |     |     |     |     | changes                                                 | in blood | flow | as the | neurovascular |     | system | responds | to  |
2
resourcedemandsofactiveneurons.Consequently,thereisadelay
1
betweenneuralactivityandthemeasuredBOLDsignal.Toaccount
|     | 0   |      |                     |           |      |      | forthis,weuseastandardhemodynamicresponsefunction, |      |        |     |       |     |       |     |     |
| --- | --- | ---- | ------------------- | --------- | ---- | ---- | -------------------------------------------------- | ---- | ------ | --- | ----- | --- | ----- | --- | --- |
|     | 0   | 1000 | 2000                | 3000 4000 | 5000 | 6000 |                                                    |      |        |     |       |     |       |     |     |
|     |     |      | Time (milliseconds) |           |      |      |                                                    | tn−1 |        |    | t    |     |       |     |     |
|     |     |      |                     |           |      |      | ( )=                                               |      |        | −   | , λ=1 | .   | , n=4 | ,   |     |
|     |     |      |                     |           |      |      | HRF t                                              | λn(  | )! Exp |     | λ     | 3s  |       |     |     |
n−1
| Fig.      | 5. DNF-model-based |          | LFPs computed | for two            | fields in Model | 2: feature      |             |     |          |          |         |     |        |      |         |
| --------- | ------------------ | -------- | ------------- | ------------------ | --------------- | --------------- | ----------- | --- | -------- | -------- | ------- | --- | ------ | ---- | ------- |
| attention | (fAtn;             | in blue) | and go node   | (green). Different | fields          | drive different |             |     |          |          |         |     |        |      |         |
|           |                    |          |               |                    |                 |                 | to describe | the | expected | response | pattern |     | in the | BOLD | signal, |
responsepatterns.Theyarecomputedunderthefollowingconditions:(A)Three
|     |     |     |     |     |     |     | for a given | amount | a   | neural | activity. | By convolving |     | HRF ( t | ) with |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | --- | ------ | --------- | ------------- | --- | ------- | ------ |
repetitions(1500mslongeach)ofLoad4,Gotrials,and(B)Sequenceoffourtrials
|     |     |     |     |     |     |     |     |     |     |     | ( ˆ ( )) |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- |
atLoad4withorderGo–Nogo–Go–Nogo.Thevariancebetweentherepetitions the long-form LFP templates LFP t , we are able to generate
isaconsequenceofthestochasticnatureofthemodel.(Forinterpretationofthe
predictedBOLDactivitypatternsthataredirectlycomparableto
referencestocolorinthisfigurelegend,thereaderisreferredtothewebversionof
themeasureddata.
| thisarticle.) |     |     |     |     |     |     |                           |     |     |     | ( )     | ˆ P(t)hasdifferentunits, |     |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | --- | ------- | ------------------------ | --- | --- | --- |
|               |     |     |     |     |     |     | NotethattimevariableinHRF |     |     |     | t andLF |                          |     |     |     |
seconds(former)andmilliseconds(latter).Also,notethatweused
5.5
|     |     |     |     |     |     |     | a mapping | of 1 | model | time-step | to  | 1 ms | in the | experiment | to  |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---- | ----- | --------- | --- | ---- | ------ | ---------- | --- |
5 simulate the details of each trial. Thus, care should be taken to
|     | 4.5 |     |     |     |     |     | bringthesetimeunitsonthesamescale,beforetheconvolution |        |                                        |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------ | ------ | -------------------------------------- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | ( )=(                                                  | HRF∗LF | ˆ )(t)iscomputed.Fig.8showstwoexamples |     |     |     |     |     |     |
|     | 4   |     |     |     |     |     | BOLD t                                                 |        | P                                      |     |     |     |     |     |     |
ytivitcA dleiF
|     | 3.5 |     |     |     |     |     | ofBOLDpredictionsobtainedasdescribedabove. |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
Next,weaddressthequestionofcomparingmodelunitsforthe
3
numericallygeneratedBOLDsignaltothosederivedfromthefMRI
2.5
data.WeagaintakeguidancefromthetreatmentoffMRIdata:we
2 normalize each predicted BOLD signal by its average value over
1.5 time across the entire experiment-length time series. This takes
usawayfrommodel-basedunitstoanabstractpercentagescale
1
relativetothemean.
0.5
0 500 1000 1500 2000 2500 3000 Then we turn these normalized BOLD signal predictions into
Time (milliseconds)
regressorsforthestatisticalanalysisofthefMRIdata.Careshould
|      |              |           |               |         |                 |            | be taken | at this | step, | again, | given that | the | calculations | require |     |
| ---- | ------------ | --------- | ------------- | ------- | --------------- | ---------- | -------- | ------- | ----- | ------ | ---------- | --- | ------------ | ------- | --- |
| Fig. | 6. Canonical | predicted | LFPs computed | for two | fields in Model | 2: feature |          |         |       |        |            |     |              |         |     |
matchingthesamplingrateofthetimeseriestothatofthedata
| attention | (fAtn; | in blue) | and go node | (green). Different | fields | drive different |     |     |     |     |     |     |     |     |     |
| --------- | ------ | -------- | ----------- | ------------------ | ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(downsamplingtomatchthetemporalresolution(TR)fromthe
responsepatterns.Theyarecomputedunderthefollowingconditions:(left,first
|      |          |       |                     |               |         |                   | fMRI data). | Fig. | 9 shows | the | normalized | BOLD | signals | resulting |     |
| ---- | -------- | ----- | ------------------- | ------------- | ------- | ----------------- | ----------- | ---- | ------- | --- | ---------- | ---- | ------- | --------- | --- |
| 1500 | ms) Load | 4, Go | trials, and (right, | last 1500 ms) | Load 4, | Nogo trials. (For |             |      |         |     |            |      |         |           |     |
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferred from those shown in Fig. 8, as well as the discrete sequence of
tothewebversionofthisarticle.) pointsretainedfromthenumericallygeneratedBOLDsignalafter
downsampling.
modelintheabsenceofexternalstimuli.Weaveragetheseread- NotethatintheanalysisoftheGnGtask,wedecidedtocreate
ings(acrossalltimepointsandrepetitions)toobtainanaverage split regressors for Go and Nogo trials (see following section for
restingvalue.Then,thisvalueissubtractedoutofourpredictions details). To split the trials, two long-form LFPs (again, for each
toexpressthechangeinLFPactivityrelativetotherestingvalue. subject and each component) were created based on only Go or
Once we have calculated a canonical baselined LFP for each Nogotrialonsetsinsteadofalltrialstogether.Theproceedingsteps
model component and condition type, we proceed to construct fromlong-formLFPtoregressorfollowidentically.

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 223
4.5
4
3.5
3
2.5
2
ytivitcA
dleiF
1.5
1
0.5
0
-0.5
0 5 10 15 20 25
Time (seconds)
Fig.7. Excerptedlong-formLFPtemplatescomputedfortwofieldsinModel2:fAtn(blue)andgonode(green).DepictedisanexperimentalblockoffourtrialsatLoad4,
presentedtoaparticularsubjectintheorderedsequenceGo–Nogo–Go–Nogo.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothe
webversionofthisarticle.)
1000
900
800
700
600
500
htgnertS
langiS
400
300
200
100
0
0 5 10 15 20 25 30 35 40 45 50
Time (seconds)
Fig.8. ExcerptedBOLDpredictionscomputedfortwofieldsinModel2:fAtn(blue)andgonode(green).SamestartingtimepointasinFig.7wasused.Depictedisasequence
ofseventrialsatLoad4withorderGo–Nogo–Go–Nogo–Go–NogoNogo.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtotheweb
versionofthisarticle.)
1.8
1.6
1.4
1.2
1
)naem
%(
htgnerts
langiS
0.8
0.6
0.4
0.2
0 5 10 15 20 25 30 35 40 45 50
Time (seconds)
Fig.9. ExcerptednormalizedanddownsampledBOLDpredictionscomputedfortwofieldsinModel2:fAtn(blue)andgonode(green).Circlesindicatethe2-second
resolutionusedtomatchthefMRITR.ThetimerangeisthesameasinFig.8.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothe
webversionofthisarticle.)
7. Testingmodel-basedpredictionswithGLM whethertheseindividually-tailoredhemodynamicpredictionsare,
infact,goodpredictionsrelativetothefMRIdatafromeachindi-
Intheprevioussection,wegeneratedalinkinghypothesisthat vidual.
allowsustospecifyalocal-fieldpotentialforeachfieldinady- We used GLM to evaluate this question. In particular, we
namicneuralfieldmodel.Wealsodetailedthestepsrequiredto usedtheindividually-tailoredhemodynamicpredictionsdescribed
transformtheseLFPsintohemodynamicpredictionsthataretai- above as regressors in a GLM for each individual participant’s
loredtoeachindividualparticipant.Thenextstepistoevaluate fMRIdata.Thisprovidesquantitativemetricswithwhichwecan

224 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
Fig.10. TestingDNFmodelpredictionswithGLM(numericalresultsusingModel2):(A)AverageHDRandLFPforGo(blue/cyan)andNogo(green/red)Load4trialsfor
thefAtnfieldandgonode.(B)PredictionsforfivecomponentsofDNFmodel(fAtn,con,wm,go,nogo)acrossLoadandProportionmanipulations;barsshowsignalchange.
(C)DNFregressorsofasinglesubjectandasamplingofthenogonode’stimecourse(atright).(Forinterpretationofthereferencestocolorinthisfigurelegend,thereader
isreferredtothewebversionofthisarticle.)
evaluate the model’sgoodness of fit. In particular, we examined maskthatdefinedvoxelswheretheDNF-basedGLMandtheStan-
thefollowingmetricsfromeachindividualGLM:(1)thenumber dardGLManalysisbothcapturedasignificantproportionofvari-
ofvoxelswherethemodel-basedGLMcapturedasignificantpro- anceandthenstatisticallycomparedtheseintersectionR2values.
portionofvariance,and(2)theaverageR2valueacrossallsignif- Thisprovidesadirecthead-to-headcomparisonofthetwomod-
icantvoxels.Notethat,becausetheR2 valueswerenotnormally elsinthesamevoxels,askingwhichmodeldoesabetterjobfitting
distributed, we z-transformed the data. An average z-value was thebraindata.Ourobjectivewastoseewhetherwecouldtunethe
calculated across the mask of voxels that were significant. The DNFmodelparameterssuchthatitsignificantlyoutperformedthe
z-transformation was then undone using R = atanh ( z ) , where StandardGLManalysisonthiscomparisonmetric.
zistheaveragez-value.Finally,theR-valuewasadjustedusing Westruggledwithtwofinalissues.First,thedegreesoffreedom
( 1−R )( N−1 ) of the DNF-based GLM and Standard GLM analysis were not the
adjR=1− same.TheStandardGLManalysisofdatafromWijeakumaretal.
N−p−1
(2015)had10regressors:5conditions(Proportion75%,Proportion
whereN =numberoftimepointsacrossrunsandp=1. 25%,Load2,Load4,Load6)×2trialtypes(Go,Nogo).Bycontrast,
AlthoughtheGLMapproachgivesusquantitativemetrics,we the DNF model had 7 regressors – one for each component (vis,
need a way to assess whether the fit of the model is any good. sAtn, fAtn, con, wm, go, nogo; see, for instance, Fig. 9) – see
As Turner et al. discuss, the optimal approach here would be to Section6forthestepsleadinguptothecreationofregressorsfrom
quantitatively compare the fit of the DNF model relative to a the DNF components. Second, we discovered when running the
competingmodel(Turneretal.,2016).Forinstance,inBussetal., DNF-basedGLMthatseveralregressorswerecollinearwhichcan
they compared hemodynamic predictions of the DNF model to makebetaestimatesunstable.Thiswasnotterriblysurprising:the
hemodynamic predictions of ACT-R (Buss et al., 2013). Here, we mostcollinearfieldswerevis,sAtn,andfAtn,andallthreefields
pursue an alternative approach that was motivated by a recent basicallyservethesamefunctionintheGnGtask.
model-basedfMRIstudyofVWM.Inthatstudy,wedidnothave To resolve both issues, we created a 10-regressor DNF-based
asecondcognitivemodelfromwhichtogeneratecompetingfMRI GLMmodelby(1)reducingthenumberofmodelcomponentsto
predictions. Instead, we compared the GLM-based fit of a DNF the5leastcollinearfields(fAtn,con,wm,go,nogo),and(2)includ-
model to Standard GLM fMRI analyses. This is useful because, ingseparatemodel-basedregressorforGoandNogotrials.
at present, Standard GLM fMRI analyses are the gold standard Fig.10illustratestheDNF-basedGLMapproachwithnumerical
in the functional neuroimaging literature and such analyses can results from Model 2 Fig. 10(A) shows examples of HDRs and
beperformedinallcases.Thus,wecantreattheStandardGLM LFPs for Load 4 Go and Nogo trials in the fAtn field and go
analysis as a baseline and ask whether the DNF-based GLM node—thesamefieldsusedforillustrationinFigs.5–9.Asabove,
quantitativelyoutperformsthisbaseline. differencesintheHDRamplitudebetweenGoandNogotrialsare
Thenextquestionis,ofcourse,whichmetrictouse.Oneop- evidentinthegonodebutnotinthefAtnfield.MaximumHDRs
tionistoanalyzevoxelcounts;however,severalstudieshavehigh- acrossthefiveDNFcomponentsincludedintheGLM(fAtn,con,
lighted the limitations of this approach (Bennett & Miller, 2010; wm,go,nogo)andacrossLoadandProportionmanipulationsare
Cohen&DuBois,1999)AnalternativeistocomparethemeanR2 displayedinFig.10(B).Thesebarsrevealdifferencesinthemodel-
valuesacrossmodels.TheproblemhereisthattheDNF-basedGLM based predictions across components and conditions. Note, for
mightcapturesignificantvarianceinsomevoxels,whiletheStan- instance, that fAtn shows comparable hemodynamic predictions
dardGLManalysismightcapturesignificantvarianceindifferent across go and nogo trials, while the go and nogo nodes show
voxels. The overall mean R2 value does not take this into effect. different patterns with, for instance, greater activation in the
Thus,weusedanalternativeapproach:wecreatedanintersection Prop25conditionongotrials,andgreateractivationintheProp75

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 225
Fig.11. Excerptsfromthe10-regressordesignmatricesforonesubjectfromthethreeGLMsfromtheproject.TheexcerptsaretakenfrompartoftheLoad6andLoad4
experimentalblocksforthegivensubject.Notethatdifferencesexistinthemodelregressorsbetweencomponents,buttheyaredifficulttoappreciateatthisscale/resolution.
conditiononnogotrials.Thisreflectsoneofthekeyhemodynamic withtheStandardGLManalysismethod.Aswell,themodel-based
patterns evident in the fMRI data: some brain areas showed a methodemploysdifferentpredictionsforeachmodelcomponent,
strongresponseoninfrequenttrials,regardlessofwhetherthose allowingustoidentifyeffectsindicativeofspecificfunctions.
trialsrequiredinhibition(anogotrialintheProp75condition)or Ineachcase,wereportthetotalnumberofsignificantvoxels
not(agotrialintheProp25condition). andthemeanR2 valueacrossthosevoxels(seebelow).Wethen
Fig.10(C)showsgoandNogotrialregressorsforeachcompo- intersectedtheimagesasperthemodelpairsandidentifiedvoxels
nentofthemodel,constructedbyinsertingthecondition-specific that were significant for both Model 1 and the Standard GLM
HDRattheonsetofeachtrialinthesameorderthatwaspresented analysis,andvoxelsthatweresignificantforbothModel2andthe
toeachparticipant.Anexamplepredictorforoneparticipant–a StandardGLManalysis.Then,wecalculatedthemeanintersection
regressorintheGLMmodel–isshownintheinsetinFig.10(C). R2 valueforeachmodelforeachparticipantandcomparedthese
This time course was created by inserting the predicted hemo- valuesusingapaired-samplest-test.
dynamictimecoursefromtheNogocomponent(similartothose Overallvoxelcountsacrossmodelswerethefollowing:Model
from Fig. 10(A)) for each trial type at the appropriate start time 1 = 3964 voxels, Model 2 = 4762, Standard GLM analysis =
inthetimeseriesandthensummingthesepredictions.Ifthereis 3978voxels.Overall,bothmodelswerecomparablebutModel2
abrainregioninvolvedinthegenerationofaNogodecision,the capturedsignificantvarianceinmorevoxelsTheoverallR2values
modelpredictsthatthisbrainareashouldshowtheparticularpat- werethefollowing:Model1=0.139,Model2=0.135,Standard
ternofBOLDchangesovertimeshownintheinset.TheGLMresults GLM analysis = 0.130, so both DNF models captured more
canbeusedtostatisticallyevaluatesuchpredictions. variance, though neither represents a significant improvement
relative to the Standard GLM analysis when we compare the
8. Modelevaluation:Individual-levelGLMs averagevaluescomputedacrossallvoxels(p=0 . 20andp=0 . 43,
respectively).
We ran 3 sets of GLM models (using afni_proc in AFNI) for TheimportantmetricinthisevaluationbetweentheDNF-based
each participant: a 10-regressor DNF-based GLM for Model 1; a GLMandtheStandardGLManalysisistheintersectionR2 values
10-regressor DNF-based GLM for Model 2; and a 10-regressor across model pairs. The intersection R2 was 0.153 for Model 1
StandardGLManalysis.AllGLManalysisalsoincludedregressors and0.141fortheStandardGLManalysisacross1616intersected
formotionanddriftsinbaseline.Fig.11showsportionsofthe10 voxels;Model1performedbetterthantheStandardGLManalysis
regressordesignmatricesfromthethreemodelsweinvestigated. but this effect did not reach significance ( t ( 19 ) = 0 . 199 , p =
Note in particular that the Standard GLM analysis employs a 0 . 086 ) .Ontheotherhand,theintersectionR2was0.150forModel
separate regressor for each trial type and condition. In contrast, 2and0.131fortheStandardGLManalysisacross1507intersected
theDNFmodel-basedmethodonlyseparatestrialsbasedontrial voxels, with Model 2 performing significantly better than the
type(goandNogotrials).Forthisreason,themodel-basedmethod Standard GLM analysis ( t ( 19 ) = 0 . 427 , p = 0 . 006 ) . When
generatesmoreconstrainedpredictionsbecausetherelationship bothDNFmodelswerecomparedagainsteachother,intersection
between trial conditions (variations in Load and Proportion) is R2 values across 1615 intersected voxels were not significantly
determined a priori and not allowed to vary independently as different,butModel2performedquantitativelybetterthanModel

226 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
1(Model1=0.148andModel2=0.149,t = 0 . 01 , p = 0 . 18). Table1
Insummary,Model2significantlyoutperformstheStandardGLM Voxel count of unique and common effects between the DNF-based GLM and
StandardGLManalysisactivationmaps.
analysisandquantitativelyperformsbetterthanModel1.Thus,at
thegrouplevelanalysis,weonlycomparedresultsbetweenModel Voxeltype Voxelcount
2andtheStandardGLManalysis. Typemaineffectonly 2610
Othereffectsonly 414
DNFcomponentsonly 2053
9. Modelevaluation:Group-levelGLMs
DNFcomponentsandtypemaineffects 494
DNFcomponentsandothereffects 97
9.1. Overviewoftheapproach
that showed combined effects from more than one component
ThebetamapsfromtheStandardGLManalysiswereinputinto
accountedfor965voxels.Importantly,allofthesevoxelsincluded
two 2-factor ANOVAs, a Load ANOVA and a Proportion ANOVA
acommonwmcomponent.Lookingatthemodelpredictionsfrom
(run using 3dMVM). The Load ANOVA consisted of Type and
Fig. 10(B), two patterns likely explain the predominance of the
LoadasfactorsandtheProportionANOVAconsistedofTypeand
wm field predictions: (1) there is a reduction in wm activation
Proportionasfactors.Themaineffectandinteractionmapsfrom
as Load was increased, and (2) there is a larger modulation of
both sets of ANOVAs were thresholded and clustered based on
family-wise corrections obtained from 3dClustSim ( α = 0 . 05). wm activation across the Proportion manipulation on Go trials
relative to Nogo trials As discussed in Wijeakumar et al. (2015),
The main effect of Type from the Proportion and Load ANOVAs
bothpatternswerepervasiveinthefMRIdata.
werepooledtogetherandcalledthe‘Typemaineffect’image.The
The DNF-based GLM approach also identified regions that
‘Othereffects’imageconsistedofthepooledeffectsfromtheLoad
laid outside of the network obtained from the Standard GLM
maineffect,Proportionmaineffect,LoadxTypeinteraction,and
Proportion×Typeinteraction. analysesapproach.Thewmfieldrecruitedpartsoftheleftfusiform
gyrus, left cuneus and left superior temporal gyrus. The lingual
The DNF-based GLM (Model 2 only) also yielded betamaps
gyrus and fusiform gyrus also reflected neural predictions of a
for each of the ten regressors. These betamaps were input into
combination of the wm, go, and nogo fields. This is consistent
an ANOVA with regressor as the only factor. The main effect of
with previous findings suggesting that the lingual gyrus plays a
regressorobtainedfromthisANOVAwascorrectedforfamilywise
role in visual memory as well as visual classification decisions
errorsusing3dClustSimasdescribedabove.Aone-samplet-test
(Mechelli, Humphreys, Mayall, Olson, & Price, 2000). Our results
was conducted within the spatial constraints of this clustered
alsoassignthesamefunctionalroletothefusiformgyruswhich
maineffectimagetoascertainthecontributionofeachregressor
to the main effect. These t-test results for each regressor were isfunctionallyconnectedtothelingualgyrusandplaysacentral
corrected for family wise errors again, identifying which model role in visual processing and visual comparison (Mechelli et al.,
components were significant predictors for each voxel. At this 2000).Anotherresultistherecruitmentofpartsoftheleftmiddle
point, we collapsed effects across trial type for each regressor. frontalgyrus(notshown)bythewmfieldandacombinationof
Forinstance,voxelsthatshowedaneffectofthewmfieldforGo thewmfieldandgoandnogonodes(Johnson,Hollingworthetal.,
trials and/or for Nogo trials were pooled together as wm areas. 2008;Johnsonetal.,2009;Simmeringetal.,2008)Thewmfield
Consequently, the final image consisted of voxels that showed playsaveryimportantroleofmaintainingmemorytracesinthe
unique and combined contributions from five fields in the DNF DNFmodelofVWMinadulthoodanddevelopment.Furthermore,
model—fAtn con, wm, go node and nogo node. This map was the middle frontal gyrus has been implicated to be involved in
intersectedwiththeTypeeffectandOtherEffectsmapsfromthe maintenance of goals and abstract representations during VWM
StandardGLManalysistoestablishwhetherthetwoGLManalyses processing(Aokietal.,2011;Barbey,Koenigs,&Grafman,2013;
identifiedsimilarbrainregionsandwhethereffectsineachcluster Haxby,Petit,Ungerleider,&Courtney,2000;Jonidesetal.,1998;
werecomparable. Munk et al., 2002; Pessoa, Gutierrez, Bandettini, & Ungerleider,
ItisimportanttonotethattheDNF-basedapproachnotonly 2002;Pessoa&Ungerleider,2004).
identifies where the brain responded in a way predicted by the The next question we examined was how these results from
model,butalsowhichfunction(s)operateswithinthatbrainregion. the DNF-based GLM overlapped with results from the Standard
Thus, in the section that follows, we examine the functional GLManalysis.Table1showsvoxelcountsforcommonandunique
networks identified by the DNF model and then compare the effects between these GLM results. Fig. 13 shows the spatial
spatialoverlapbetweentheDNF-basedGLMandtheStandardGLM distributionoftheseclustersfortheuniqueandcommoneffects.
analysis. TheTypemaineffectfromtheStandardGLManalysisoverlapped
with534voxelsthatwerealsosignificantintheDNF-basedGLM
9.2. Group-levelresults (Fig.13;yellow).Inaddition,the‘Othereffects’fromtheStandard
GLManalysisoverlappedwith116voxelsthatwerealsosignificant
Fig. 12 shows those DNF model predictors that produced in the DNF-based GLM (shown in brown in Fig. 13). We focus
statisticallysignificantclusterswithinthebrainregionsshowing on these overlapping effects below because they provide a way
amaineffectofcomponent.Overall,theDNF-basedGLMrevealed to evaluate our model-based fMRI results relative to findings
patternsofactivationconsistentwiththemodel-basedpredictions discussedinWijeakumaretal.(2015).
incorticalandsub-corticalnetworksofthebrainthatincludedthe Tables2and3showclustersthatoverlappedbetweentheDNF-
cerebellum,putamen,insula,caudate,supplementarymotorarea basedGLMandtheTypemaineffectandOthereffectsrespectively.
(SMA), as well as parts of the occipital cortex and the cingulate For each overlapping cluster, we identify the fields that were
cortex. significantintheDNF-basedGLM.
Uniquecontributionsfromthewmfieldrecruitedthelargest Critically, there was overlap between the areas recruited by
numbers of regions (accounting for 1738 voxels). Critically, key the wm field and the Type main effect in parts of the insular-
partsoftheinsular-thalamic-putamennetworkwereassignedto thalamic-putamen network. As noted above, this is consistent
aworkingmemoryfunction,consistentwithclaimsbyHampshire withclaimsbyHampshireandcolleaguesthatworkingmemory
andcolleagues(Erika-Florenceetal.,2014)thatworkingmemory playsacentralroleinresponseselectionviaactivationofanterior
andattentionprocessesmayunderlieresponseselection.Clusters insularandfrontaloperculumnetwork(Erika-Florenceetal.,2014;

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 227
withthewmfieldincludingportionsofthecerebellarregionsand
|     |     |     |     | also the           | insula and putamen. | In our         | previous work, | this          | insular |
| --- | --- | --- | --- | ------------------ | ------------------- | -------------- | -------------- | ------------- | ------- |
|     |     |     |     | network            | has been implicated | in detecting   | salient        | or infrequent |         |
|     |     |     |     | events (Wijeakumar | et                  | al., 2015). In | the model,     | the wm        | field   |
|     |     |     |     | is responsible     | for associating     | and retrieving | the            | appropriate   | SR      |
mappingstobothfrequentornon-salientandaswellasinfrequent,
|     |     |     |     | salient events. | As noted | above, the | wm field | showed two | key |
| --- | --- | --- | --- | --------------- | -------- | ---------- | -------- | ---------- | --- |
effectsthatwerepervasiveintheStandardGLManalysisresults:
areductioninactivationoverLoadandalargermodulationofwm
activationacrosstheProportionmanipulationonGotrialsrelative
toNogotrials.Thislikelyexplainstheoverlapbetweenpredictions
fromthewmfieldandtheOthereffects.
10. Generaldiscussion
|     |     |     |     | The objective | of the                   | current paper | was      | to formalize | an    |
| --- | --- | --- | --- | ------------- | ------------------------ | ------------- | -------- | ------------ | ----- |
|     |     |     |     | integrative   | cognitive neuroscience   | approach      | using    | DFT. To      | this  |
|     |     |     |     | effect, we    | adopted a tutorial-style |               | approach | wherein we   | first |
introducedDFTanditsapplicationstoreaderswhomightbeless
familiarwiththismodelingapproach.Then,weuseddatafroma
responseselectionparadigmasanexemplarcasestudytoexplain
|     |     |     |     | the steps     | and rationale | involved in | building DNF | models         | that |
| --- | --- | --- | --- | ------------- | ------------- | ----------- | ------------ | -------------- | ---- |
|     |     |     |     | could capture | behavioral    | and neural  | data and     | the challenges | in   |
bridgingbrainandbehaviorusingthesemethods.Thecentralgoal
Fig. 12. Functional maps generated by DNF model. Colored regions represent ofthisapproachwastogeneratehemodynamicpredictionsfrom
corticalareaswhereamaineffectofcomponentwaspresent.(Forinterpretation DNFmodelsandevaluatethesepredictionsattheindividualand
ofthereferencestocolorinthisfigurelegend,thereaderisreferredtotheweb
grouplevelsusingGLMbymakingcomparisonstoStandardGLM
versionofthisarticle.)
analyses.
TwoDNFmodelscapturedbehavioraldatafromthetaskrea-
sonablywell;however,onlyoneoftheDNFmodelsoutperformed
|     |     |     |     | the Standard | GLM analysis | when comparing | adjusted | R2  | values |
| --- | --- | --- | --- | ------------ | ------------ | -------------- | -------- | --- | ------ |
withinthesameregionsofthebrain.Interestingly,thismodelar-
chitecturewasdevelopedbytuningthefirstmodelparametersto
capturecompetitiveneuralinteractionsfirstandthensimultane-
ouslycapturingbehavioraldataaswell.Thissuggeststhatiterative
modelingusingthisapproachmightbemosteffective.Model2was
thenadvancedtothegrouplevelanalysestolookatspatialdis-
tributionsofDNFcomponentsandhowthesedistributionsover-
lappedwitheffectsobservedintheStandardGLManalysisfrom
ourpreviouswork.
|     |     |     |     | The DNF | model engaged | a large cortico-sub-cortical |     | network |     |
| --- | --- | --- | --- | ------- | ------------- | ---------------------------- | --- | ------- | --- |
thatincludedpartsofthecerebellum,SMA,insula,putamen,thala-
mus,caudateandpartsoftheoccipitalcortex.Inparticular,unique
|     |     |     |     | contributions | from the    | wm field accounted | most | of spatial      | dis- |
| --- | --- | --- | --- | ------------- | ----------- | ------------------ | ---- | --------------- | ---- |
|     |     |     |     | tributions.   | The rest of | the contributions  | were | from a combina- |      |
tionofeffectsbetweenthewmfieldandothercomponentsinthe
DNFmodel.ThisfindingisinlinewithHampshireandcolleagues
whoarguethatresponseselectionandinhibitionisapropertyof
|     |     |     |     | spatially | distributed functional | networks | that support | a general |     |
| --- | --- | --- | --- | --------- | ---------------------- | -------- | ------------ | --------- | --- |
classofworkingmemoryandattentionalprocesses(Erika-Florence
Fig.13. OverlapbetweenDNFandthestandardGLManalysis.(Forinterpretation
| ofthereferencestocolorinthisfigurelegend,thereaderisreferredtotheweb |     |     |     | etal.,2014). |     |     |     |     |     |
| -------------------------------------------------------------------- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
versionofthisarticle.) These spatial distributions also overlapped with effects from
|     |     |     |     | the Standard | GLM analysis. | Findings | from the | Cisek lab | might |
| --- | --- | --- | --- | ------------ | ------------- | -------- | -------- | --------- | ----- |
Hampshire&Sharp,2015)Overlapbetweencombinationsofthe provide some evidence that are in line with our findings on the
wm field and other components and the Type main effect was recruitment of a host of cortical and sub-cortical regions by the
alsoobservedinpartsofthecerebellumandSMA.Itisinteresting wmfieldthatoverlappedwithareasshowingadifferencebetween
|                 |             |                    |             | Go and | Nogo responses | in the Standard | GLM | analysis | (Cisek, |
| --------------- | ----------- | ------------------ | ----------- | ------ | -------------- | --------------- | --- | -------- | ------- |
| that activation | elicited by | the DNF components | seemed more |        |                |                 |     |          |         |
localizedascomparedtotheactivationfromtheTypemaineffect 2012). These authors presented evidence that action selection
(see yellow regions embedded in red regions in Fig. 13). This is emerges through a distributed consensus across many levels of
an encouraging sign for future work, suggesting that the DNF representation,whichinthecurrentcasewouldrepresentmultiple
modelmightidentifyfunctionalnetworksthataremoreprecisely SR mappings. According to this theory, cortical and subcortical
localizedthanwhatistypicallyrevealedbyStandardGLManalyses. regionscompetethroughinhibitoryinteractionswhenindividuals
The Other effects activation maps contained the effects of arefacedwithmultiplepotentialactions.So,itispossiblethatthe
ProportionandLoadandinteractionsofthesetwomanipulations BOLDsignalreductionreportedinourpreviousworkisrelatedto
withTypeoftrial.Onceagain,thegreatestdegreeofoverlapwas theinhibitorycompetitionbetweentheGoandNogoresponses.

228 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
Table2
SpatialoverlapbetweenDNFmodelandthetypemaineffectfromthestandardGLManalysis.
Components Intersectedwith Region Hemi Volume(mm3) Centerofmass
|     |        |            |     |      | x y      | z     |
| --- | ------ | ---------- | --- | ---- | -------- | ----- |
| WM  | TypeME | Putamen    | L   | 4502 | 22.6 3.1 | 10.8  |
|     | TypeME | Cerebellum | L   | 1458 | 2.6 66.5 | −30.4 |
|     |        |            |     |      | −15.6    | −33.8 |
|     | TypeME | Cerebellum | R   | 1158 | 47.7     |       |
−21.2
|     | TypeME | Caudatenucleus        | R   | 943 | 7.8        | 18.3 |
| --- | ------ | --------------------- | --- | --- | ---------- | ---- |
|     | TypeME | Superiortemporalgyrus | L   | 858 | 37.8 24.4  | 6.8  |
|     | TypeME | Putamen               | R   | 686 | −23.2 −9.1 | 4.5  |
−40.2
|     | TypeME | Cerebellum     | L   | 514 | 17.5 56.1  |       |
| --- | ------ | -------------- | --- | --- | ---------- | ----- |
|     | TypeME | Cerebellum     | R   | 429 | −28.0 64.0 | −14.2 |
|     | TypeME | Cerebellum     | R   | 386 | −34.8 51.2 | −25.8 |
|     |        |                |     |     | −17.3      | −22.7 |
|     | TypeME | Cerebellum     | R   | 386 | 53.9       |       |
|     | TypeME | Thalamus       | R   | 343 | −9.2 23.3  | 14.4  |
|     | TypeME | Cingulategyrus | R   | 343 | 6.1 21.1   | 33.6  |
|     | TypeME | SMA            | R   | 257 | −3.5 5.1   | 54.9  |
−11.0
|           | TypeME | Cerebellum               | L   | 214 | 25.6 72.4  |       |
| --------- | ------ | ------------------------ | --- | --- | ---------- | ----- |
|           | TypeME | Cerebellum               | L   | 172 | 24.5 51.8  | −46.0 |
|           | TypeME | Insula                   | R   | 172 | −36.8 1.9  | 3.0   |
|           |        |                          | −   |     |            | −0.5  |
|           | TypeME | Cerebellum               |     | 172 | 0.0 44.8   |       |
|           | TypeME | Cerebellum               | R   | 129 | −30.9 43.0 | −33.2 |
|           | TypeME | Cerebellum               | R   | 129 | −20.4 60.5 | −12.2 |
|           | TypeME | Caudatenucleus           | R   | 129 | −11.1 −3.7 | 20.5  |
|           | TypeME | Posteriorcingulatecortex | L   | 129 | 5.2 27.8   | 26.3  |
|           | TypeME | Cerebellum               | L   | 86  | 10.5 64.0  | −35.5 |
|           | TypeME | Cerebellum               | R   | 86  | −8.8 43.0  | −5.8  |
|           |        |                          |     |     | −3.5       | −7.5  |
|           | TypeME | Cerebellum               | R   | 86  | 53.5       |       |
|           | TypeME | Thalamus                 | R   | 86  | −19.2 6.2  | 3.0   |
| WMandfAtn | TypeME | Cerebellum               | L   | 129 | 7.6 69.8   | −11.0 |
−7.5
| WM,GonodeandNogonode | TypeME | FusiformGyrus | L   | 86  | 24.5 74.5 |      |
| -------------------- | ------ | ------------- | --- | --- | --------- | ---- |
|                      | TypeME | SMA           | L   | 86  | 12.2 6.2  | 55.5 |
| WMandCon             | TypeME | SMA           | L   | 86  | 1.8 1.0   | 53.8 |
−21.5
| WMandGonode   | TypeME | Cerebellum | L   | 514  | 35.0 50.0  |       |
| ------------- | ------ | ---------- | --- | ---- | ---------- | ----- |
|               | TypeME | Putamen    | L   | 514  | 22.8 14.1  | 8.2   |
|               | TypeME | Putamen    | L   | 429  | 32.2 4.8   | 3.0   |
|               |        |            |     |      | −24.5      | −13.3 |
|               | TypeME | Cerebellum | R   | 257  | 71.0       |       |
|               | TypeME | Cerebellum | R   | 129  | −41.4 51.2 | −21.5 |
| WMandNogonode | TypeME | Cerebellum | R   | 2187 | −25.6 48.9 | −17.1 |
−22.2
|     | TypeME | Cerebellum | L   | 686 | 17.7 59.6  |       |
| --- | ------ | ---------- | --- | --- | ---------- | ----- |
|     | TypeME | Cerebellum | R   | 557 | −7.7 62.4  | −30.9 |
|     | TypeME | Cerebellum | R   | 514 | −9.3 25.5  | −19.5 |
|     | TypeME | Cerebellum | R   | 343 | −16.6 53.9 | −42.1 |
|     |        |            |     |     | −13.7      | −6.8  |
|     | TypeME | Cerebellum | R   | 214 | 50.0       |       |
|     | TypeME | Cerebellum | L   | 172 | 41.1 51.8  | −18.9 |
|     | TypeME | Cerebellum | L   | 172 | 24.5 58.8  | −11.0 |
|     |        |            |     |     | −18.1      | −21.5 |
|     | TypeME | Cerebellum | R   | 129 | 34.8       |       |
|     | TypeME | Cerebellum | L   | 86  | −12.2 64.0 | −19.8 |
Table3
SpatialoverlapbetweentheDNFmodelandtheOthereffectsfromthestandardGLManalysis.
Volume(mm3)
| Components | Intersectedwith | Region                 | Hemi |     | Centerofmass |      |
| ---------- | --------------- | ---------------------- | ---- | --- | ------------ | ---- |
|            |                 |                        |      |     | x y          | z    |
|            |                 |                        |      |     | −39.8        | −6.2 |
| WM         | Othereffects    | Inferioroccipitalgyrus | R    | 943 | 69.1         |      |
|            | Othereffects    | Cerebellum             | R    | 557 | −18.4 54     | −28  |
|            | Othereffects    | Putamen                | R    | 557 | −26 −0.3     | 4.9  |
|            | Othereffects    | Middletemporalgyrus    | R    | 386 | −44.1 51.2   | 6.1  |
−8.5
|     | Othereffects | Cerebellum     | L   | 300 | 7.8 50      |       |
| --- | ------------ | -------------- | --- | --- | ----------- | ----- |
|     | Othereffects | Cerebellum     | L   | 257 | 1.8 48.2    | −18   |
|     | Othereffects | Insula         | R   | 172 | −36.8 −12.1 | 6.5   |
|     |              |                |     |     | −35.6       | −15.7 |
|     | Othereffects | Fusiformgyrus  | R   | 129 | 50          |       |
|     | Othereffects | Putamen        | R   | 86  | −28 −9.5    | −4    |
|     | Othereffects | Insula         | R   | 86  | −33.2 −18.2 | 6.5   |
|     | Othereffects | Caudatenucleus | L   | 86  | 8.8 4.5     | 15.2  |
|     |              |                |     |     | −38.5       | −4    |
WM,fAtnandGonode Othereffects Inferioroccipitalgyrus R 86 74.5
WM,GonodeandNogonode Othereffects Lingualgyrus L 172 13.1 64 −4
−7.5
|               | Othereffects | Cerebellum    | L   | 86  | 17.5 67.5  |      |
| ------------- | ------------ | ------------- | --- | --- | ---------- | ---- |
| WMandGonode   | Othereffects | Fusiformgyrus | R   | 86  | −26.2 65.8 | −4   |
| WMandNogonode | Othereffects | Cerebellum    | R   | 86  | −14 64     | −25  |
|               |              |               |     |     | −8.8       | −0.5 |
|               | Othereffects | Cerebellum    | R   | 86  | 55.2       |      |

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 229
The wm field also engaged regions in the occipital cortex, severalfieldsandparametersthatrequirefine-tuningtogenerate
an insular ‘salience’ network and the cerebellum. Collectively goodfitstobothbehavioralandneuraldata.However,thisadded
taken, we suggest that wm field is involved in processing visual level of complexity is to be expected if one tries to bridge
non-linearpatternsofbrainactivityandmacroscopicbehavioral
| information | from | the | stimuli, | to associating |     | and retrieving | the |     |     |     |     |     |     |     |     |
| ----------- | ---- | --- | -------- | -------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
appropriateSRmappingstobothfrequentornon-salientand as responses.Wecontendthatbridgingbrainandbehaviorrequires
well as salient events, before activating the motor planning and modelsthattakeintoaccounthowneuralsystemsactuallywork.
execution centers of the brain. These findings show a departure DFT does this by faithfully capturing many known properties
ofourDNFmodelfromtypicalintegrativemodelingapproaches, of neural population dynamics and how neural populations are
asemphasizedbyTurnerandcolleagues(Turneretal.,2016).As recurrently connected across multiple cortical fields to give rise
these researchers underline, integrative models require a strong tocomplexbehaviors(Bastianetal.,1998,2003;Erlhagenetal.,
| commitmenttoboththeunderlyingcognitiveprocessandwhere |     |     |     |     |     |     |     | 1999). |     |     |     |     |     |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
thisprocessisexecutedinthebrain.TheDNFmodeldoesnotfall Thatsaid,itisalsoimportanttonotethatDFTdoesnotconsider
intothiscategory.TheDNFmodeldoesshowastrongcommitment other known aspects of neural function such as the details of
tospecifyingthecognitiveandneuralprocessesthatunderliethe neurotransmitter action, the biophysical properties of individual
behaviors in questions; however, our approach remains open to neurons,andsoon(Garagnani,Wennekers,&Pulvermüller,2008;
Markrametal.,2015).Inthissense,DFTprovidesalimitedviewof
whereinthebraintheseneuraldynamicslive.Thisisanimportant
neuralfunction.Totheextentthatthesedetailsmatter,evenmore
observation—remember,neuronsdonotalwaysactlikemodules.
complexbiophysicalmodelswillberequiredifwewanttobridge
| Neurons | can switch | their | allegiance, | thus | coding | for | multiple |     |     |     |     |     |     |     |     |
| ------- | ---------- | ----- | ----------- | ---- | ------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
brainandbehavior.Ourclaim,however,isthatmanyoftheselow-
dimensions.Soallowingforflexibilityintheintegrativemodeling
|     |     |     |     |     |     |     |     | level biophysical |     | details | are not | necessary | when | capturing | fMRI |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------- | ------- | --------- | ---- | --------- | ---- |
approachmaybebeneficialwhenmappingtheoriestocognitive
processesinthebrain.Inthenextsection,wecriticallyevaluate databecausefMRIprovidesonalow-passfilteronneuralactivity.
this modeling approach with an eye toward future efforts to Futureworkwillbeneededtoevaluatethisconjecture.Critically,
optimizemodelperformanceandfurtherDFTapplications. however, the approach described here facilitates that work by
|     |     |     |     |     |     |     |     | providing | a formal | method | to  | test whether | neural |     | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------- | ------ | --- | ------------ | ------ | --- | ---------- |
dynamicsaresufficienttocapturethedetailsinherentinfMRI.
10.1. Evaluatingthemodel-basedapproach
|                |            |                  |               |               |                |               |             | Exploratory  | versus  | confirmatory   |             | modeling       | approaches:  |                | Turner      |
| -------------- | ---------- | ---------------- | ------------- | ------------- | -------------- | ------------- | ----------- | ------------ | ------- | -------------- | ----------- | -------------- | ------------ | -------------- | ----------- |
|                |            |                  |               |               |                |               |             | et al. argue | that    | integrative    | models      | are            | confirmatory |                | by nature   |
| This tutorial  |            | has meticulously |               | walked        | through        |               | explaining  |              |         |                |             |                |              |                |             |
|                |            |                  |               |               |                |               |             | because      | fits to | brain networks |             | and behavioral |              | patterns       | are         |
| the background |            | to DFT,          | previous      | applications, |                | the rationale | for         |              |         |                |             |                |              |                |             |
|                |            |                  |               |               |                |               |             | constrained. | We      | agree with     | this        | outlook        | (Turner      | et             | al., 2016). |
| developing     | DNF        | models,          | construction  | of            | the components |               | of the      |              |         |                |             |                |              |                |             |
|                |            |                  |               |               |                |               |             | However,     | in the  | current        | case study, | there          | is also      | an exploratory |             |
| fields of      | DNF models |                  | and comparing |               | quantitative   |               | fits to the |              |         |                |             |                |              |                |             |
|                |            |                  |               |               |                |               |             | component.   | For     | instance,      | one of      | our central    | questions    |                | here was    |
behavioralandneuraldatatoStandardGLManalysesThisraised
|                |     |           |     |             |         |              |     | exploratory | in  | nature: can | components |     | from | previous | working |
| -------------- | --- | --------- | --- | ----------- | ------- | ------------ | --- | ----------- | --- | ----------- | ---------- | --- | ---- | -------- | ------- |
| several issues | we  | summarize |     | here in our | efforts | to formalize | an  |             |     |             |            |     |      |          |         |
memorymodelscapturebrainandbehavioralpatternsinresponse
integrativecognitiveneuroscienceapproach.
selection?Oncewehaveamodelthatdoesthis,wecanmoveinto
Choosing parameters for DNF models: We obtained reasonable theconfirmatoryphase.Arefinedapproachatthisstagewouldbe
behavioral fits for both DNF models using parameters grounded todesignconditionsinthetaskthatde-correlatethefieldsofthe
bypreviouswork(Erlhagen&Schöner,2002)andourexperience
DNFmodelForinstance,ifwefindthatdecreasingtheproportion
| with learning      | dynamics. |       | That    | said, it is | possible     | that | different  |              |          |              |     |              |     |        |          |
| ------------------ | --------- | ----- | ------- | ----------- | ------------ | ---- | ---------- | ------------ | -------- | ------------ | --- | ------------ | --- | ------ | -------- |
|                    |           |       |         |             |              |      |            | of go trials | resulted | in different |     | LFP patterns | in  | the wm | field as |
| sets of parameters |           | could | provide | similar     | quantitative |      | behavioral |              |          |              |     |              |     |        |          |
comparedtothegonode,thenarangeofproportionoftrialscanbe
| fits. Future | work | will | be needed | to explore | a   | broader | range of |     |     |     |     |     |     |     |     |
| ------------ | ---- | ---- | --------- | ---------- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
testedtodeterminethepointatwhichcollinearitybetweenthose
| parameters, | asking | two | key questions: |     | (1) are | there parameters |     |     |     |     |     |     |     |     |     |
| ----------- | ------ | --- | -------------- | --- | ------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tworegressorswouldbeatthelowest,whilststillpreservingthe
| that provide | a   | better | fit to | the behavioral |     | and neural | data, |     |     |     |     |     |     |     |     |
| ------------ | --- | ------ | ------ | -------------- | --- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
integrityoftheDNFmodel.Further,onecouldtesttheefficiencyof
| and (2) do | we  | see the | same | qualitative | behavioral |     | and neural |     |     |     |     |     |     |     |     |
| ---------- | --- | ------- | ---- | ----------- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
multipledesignmatricesconstructedfromsuchregressors.After
outcomesfromthemodelacrossarangeofparameters,without
thisconfirmatoryphase,onecouldoptimallytestthemodelacross
| dramatic | violations | of  | the behavioral | and | neural | patterns. | The |         |               |         |     |                |     |        |          |
| -------- | ---------- | --- | -------------- | --- | ------ | --------- | --- | ------- | ------------- | ------- | --- | -------------- | --- | ------ | -------- |
|          |            |     |                |     |        |           |     | a range | of scenarios. | Indeed, | the | ideal scenario |     | is one | in which |
former question examines the goodness-of-fit of the model; the theconfirmatoryphaseenablescontrastswithothertheoriesthat
latter question probes the generality of the model. We think an makedifferentpredictionsforbothbrainandbehavior.
iterative approach to model exploration would be most fruitful Wenote,however,thatdoingthisrequireshavingcomparable
here,stressingtheimportantconstraintsgainedbymodelingtwo theoretical approaches such as two integrative cognitive neuro-
datasetssimultaneouslyfromasingleneuralprocessmodel. sciencemodels.Atpresent,thisisdifficultgiventhattherearerela-
Constrainingthemodel:Despitenottestingamultitudeofpa- tivelyfewintegrativeapproaches(butsee,(Bussetal.,2013)).One
rameters, there are still many points in this modeling approach alternativeistocontrasttwodifferentmodelsfromthesamethe-
whereconstraintshavebeenplaced.Tobegin,thearchitecturewas oreticalframework.Wedidavariantofthisinthecurrentstudy,
heavilyconstrainedbyusingcomponentsthathaveahistoryinex-
contrastingModel1withModel2.Amoreconceptuallyintrigu-
| plaining working |     | memory | processes | (Johnsonet |     | al., 2009; | John- |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------ | --------- | ---------- | --- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
ingvariantofthisapproachwouldbetocontrasttwodifferentdy-
son,Spencer,&Schöner,2008;Simmering&Spencer,2007).This
namicfieldarchitectures(ratherthantestingthesamearchitecture
wasdonetoplaceemphasisonthegeneralizationofthesecom-
underdifferentparametersettings).Whencontrastedatthelevels
ponentsacrossdifferentexecutivefunctions.Next,weconstrained ofbothbrainandbehavior,thismightenableonetoeliminatecan-
the model to account for both behavioral and neural data—the didatemodelsbasedonthefittodata.
keystrengthofadoptinganintegrativecognitiveneuroscienceap- Difficultyofimplementation:Developingadynamicfieldmodel
proach. Concretely, constraints here come from the direct map- and fitting the model to data is a complex enterprise. However,
pingofneuralactivationpatternsinthemodeltoLFPstosimulated therecentbookfromtheDFTgroupunpacksthiscomplexity,pro-
BOLDdata.Finally,infuturework,constraintscanalsobeapplied vidingthebackgroundtoDFTincludingtheunderlyingrationale.
whenmappingfromonemodeltothenextwithagoaltointegrate The book also offers multiple examples of implemented models
acrossDNFarchitectures. thatcanhelpfosterthedevelopmentofnewmodels.Further,the
Model Complexity. When contrasted with other cognitive COSIVINAsimulationenvironmentallowsresearcherstobuilden-
models,DNFmodelsseemrathercomplex.Theyarecomposedof tireDFmodelsusingafewlinesofcodemakingimplementation

230 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
easy.Wenotethatwehaveaddedaneuroimagingtoolboxtothis Westartbydescribingtheequationforthevisualfield.Besides
framework;thus,creatingtheLFPsdescribedhereinisquiteeasy local neuronal population interactions, the visual field receives
(seewww.dynamicfieldtheory.org/software/). excitatoryconnectionsfromthespatialattentionandthefeature
|     |     |     |     |     |     |     |     |     |     |     |     |     | ∗g  | (u )andcvis,fAtn | ∗   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- |
Uncovering the ‘ground truth’ amongst models: An important attentionfieldsviaconvolutionscvis,sAtn sAtn sAtn
|     |     |     |     |     |     |     |     |    |    |     |     |     |     | (x  | , y). |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
issue to address in future work would be the nature of spatial g u .Itisalsosubjecttoexternalstimulussvis
sAtn fAtn
neuralpatternsinthecortexthatarerevealedbytheDNF-based τ ( , , )=−uvis ( , , )+hvis
|                                                           |     |     |     |     |     |     |     | u˙ vis | x y t |     | x y t |     |     |     |     |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------ | ----- | --- | ----- | --- | --- | --- | --- |
| approachrelativetoStandardGLManalyses.Mostcritically,when |     |     |     |     |     |     |     | e      |       |     |       |     |     |     |     |

thetwoapproachesdisagree,whichapproachrevealsthe‘ground +  x−x ′, y−y ′   ′, ′,  ′ ′
|     |     |     |     |     |     |     |     |     | cvis |     |     | gvis uvis | x y | t dx dy |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --------- | --- | ------- | --- |
truth’?Oneinterestingavenuetoexplorethisquestionwouldbe
| tocarefullyintroducedifferenttypesofsyntheticdataintoanfMRI |     |     |     |     |     |     |     |     |    |       |     |     |       |     |     |
| ----------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ----- | --- | --- |
|                                                             |     |     |     |     |     |     |     | +   |     |  x−x | ′  |   | ′,  | ′   |     |
dataset.Forinstance,onecouldeffectivelyinsertneuralpatterns cvis,sAtn g sAtn u sAtn x t dx
| consistent | with | the DNF | model, | inconsistent |     | with the | model, or |     |     |     |     |     |     |     |     |
| ---------- | ---- | ------- | ------ | ------------ | --- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |

unbiased to either approach. One could then use Standard GLM + ( y−y ′)  ( ′, ) ′
|     |     |     |     |     |     |     |     |     | cvis,fAtn |     | g    | u y  | t dy |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ---- | ---- | ---- | --- | --- |
|     |     |     |     |     |     |     |     |     |           |     | fAtn | fAtn |      |     |     |
analysisandDNFapproachestofishouttheseactivationpatterns.
|     |     |     |     |     |     |     |     | +η  | (x , | , t)+svis | (x , y). |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --------- | -------- | --- | --- | --- | --- |
Inthiscase,oneknowsthe‘groundtruth’anditiseasiertoevaluate vis y (A.1)
| which method |     | outperforms | the | other. | Then | one could | explore |     |         |           |       |          |     |            |         |
| ------------ | --- | ----------- | --- | ------ | ---- | --------- | ------- | --- | ------- | --------- | ----- | -------- | --- | ---------- | ------- |
|              |     |             |     |        |      |           |         | The | spatial | attention | field | receives | two | excitatory | inputs: |
theoverlap(orlackthereof)acrossspatialdistributionsbetween
|                                               |     |     |     |     |     |     |     | projections | c    | sAtn,vis ∗ | g (uvis | ) from      | the visual | field,      | and a sub- |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ---------- | ------- | ----------- | ---------- | ----------- | ---------- |
| approachestobetterunderstandthediscrepancies. |     |     |     |     |     |     |     |             |      |            | sAtn    |             |            |             |            |
|                                               |     |     |     |     |     |     |     | threshold   | bump | activity   | s       | ( x ) . The | latter     | is centered | at the     |
Although future work in this direction will be needed, we sAtn
positionofstimuluspresentationanditsimulatestheresponseof
| note that | compared | to  | Standard | GLM | analyses, | DNF | models are |     |     |     |     |     |     |     |     |
| --------- | -------- | --- | -------- | --- | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
thenetworkduringthefixationstageofthetask
groundedinaformaltheorythatspecifieshowneuralpopulations
dynamics give rise to behavioral patterns. In this sense, the τ u˙ ( , ) = −u ( , )+h
|           |               |     |     |          |      |              |     | e sAtn | x t | sAtn | x t   | sAtn |     |           |     |
| --------- | ------------- | --- | --- | -------- | ---- | ------------ | --- | ------ | --- | ---- | ----- | ---- | --- | --------- | --- |
| fact that | the DNF-based |     | GLM | reported | here | outperformed | the |        |     |     |       |      |     |           |     |
|           |               |     |     |          |      |              |     |        |     |      |      | ′   |   |         |     |
|           |               |     |     |          |      |              |     |        |     | +    | c x−x | g    | u   | x ′, t dx | ′   |
StandardGLManalysisonkeyquantitativemetricsisimportant. sAtn sAtn sAtn
Nevertheless,werecognizethatthereisoftenaninherentmistrust 
|     |     |     |     |     |     |     |     |     |     |     |     |  ′ |    |    |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
with formal models and empirically-oriented researchers will + c x−x gvis uvis x ′, y ′, t dy ′ dx ′
sAtn,vis
| likely gravitate |     | toward | Standard | GLM | analyses | to  | provide the |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------ | -------- | --- | -------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
‘ground truth’. This is certainly a reasonable approach until the +η ( , )+s ( ).
|     |     |     |     |     |     |     |     |     |     | sAtn | x t | sAtn x |     |     | (A.2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ------ | --- | --- | ----- |
DNF-basedintegrativecognitiveneuroscienceapproachprovesits
worthacrossmultipleprojects. Thefeatureattentionfieldreceivesexcitatoryinputsfromthe
visual,contrastandworkingmemoryfields:
|                 |     |     |     |     |     |     |     | τ u˙   | ( , ) = | −u   | ( , )+h |      |     |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ---- | ------- | ---- | --- | --- | --- |
| Acknowledgments |     |     |     |     |     |     |     | e fAtn | y t     | fAtn | y t     | fAtn |     |     |     |

|     |     |     |     |     |     |     |     |     |     | +   |  y−y | ′  |  ( | ′, ) ′ |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ------- | --- |
This material is based upon work supported by the National c fAtn g fAtn u fAtn y t dy
ScienceFoundationunderGrantNumberBCS-1029082(awarded

toJPSandRC)andbytheEuniceKennedyShriverNationalInstitute +  y−y ′   ′, ′,  ′ ′
|          |        |           |     |             |       |       |        |     |     |     | c fAtn,vis |     | gvis uvis | x y t | dy dx |
| -------- | ------ | --------- | --- | ----------- | ----- | ----- | ------ | --- | --- | --- | ---------- | --- | --------- | ----- | ----- |
| of Child | Health | and Human |     | Development | under | grant | number |     |     |     |            |     |           |       |       |

| R01HD083287(awardedtoJPS). |     |     |     |     |     |     |     |     |     | +   |           | y−y ′ |    | ( ′, ) | ′   |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | --- | ------- | --- |
|                            |     |     |     |     |     |     |     |     |     |     | c fAtn,con | g      | u   | y t dy  |     |
con con
| AppendixA                                     |     |     |     |     |     |     |     |     |     |    |           |        |         |         |       |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ------- | ------- | ----- |
|                                               |     |     |     |     |     |     |     |     |     | +   |          | y−y ′ |        | ( ′, ) | ′     |
|                                               |     |     |     |     |     |     |     |     |     |     | c fAtn,wm |        | gwm uwm | y t dy  |       |
| A.1. Dynamicfield(DNF)modelforGo/Nogoparadigm |     |     |     |     |     |     |     |     |     | +η  | ( , ).    |        |         |         |       |
|                                               |     |     |     |     |     |     |     |     |     |     | y t       |        |         |         | (A.3) |
fAtn
The dynamic field (DNF) model for the Go/Nogo paradigm Thecontrastfieldreceivesfeedforwardexcitatoryconnections
consists of 7 coupled neuronal subnetworks as illustrated in fromthevisualandfeatureattentionfields;inhibitoryconnections
Fig.2:thevisualfield(vis);spatialattentionfield(sAtn);feature fromtheworkingmemoryfield;andexcitatoryfeedbackfromthe
nogonode.Toaccountforlearningduringthepre-taskinstruction
| attention | (fAtn); | contrast | field | (con); | working | memory | (wm); |     |     |     |     |     |     |     |     |
| --------- | ------- | -------- | ----- | ------ | ------- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
( )
and the ‘‘decision system’’ consisting of two nodes (go and step,asub-thresholdinputs con y withactivitybumpslocalizedat
Nogo). The DNF Model 1 is therefore defined by a system of theNogocolorsisalsoincluded
five integral–differential equations (A.1)–(A.5) and two ordinary τ u˙ ( , )=−u (y , t)+h
|                                                 |     |     |     |     |     |     |     | e con | y t | con  |     | con |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ---- | --- | --- | --- | --- | --- |
| differentialequations(A.6)–(A.7),aslistedbelow. |     |     |     |     |     |     |     |       |    |      |     |     |     |     |     |
|                                                 |     |     |     |     |     |     |     |       |     |  ′ |    | )  |     |     |     |
Eachequationisdescribedbyasumofseveralcomponents.The + c y−y g u ( y ′, t dy ′
|     |     |     |     |     |     |     |     |     | con |     | con | con |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
firstthreetermscorrespondtothelocalfieldinteractions,while

| lo ca l n o is | e i s m | o d e le d | b y t h e | f u n c ti | on η . A ll | t e r m s | th a t d e p e n d |     |     |       |     |     |          |     |     |
| -------------- | ------- | ---------- | --------- | ---------- | ----------- | --------- | ------------------ | --- | --- | ----- | --- | --- | -------- | --- | --- |
|                |         |            |           |            |             |           |                    | +   |     |  y−y | ′  |   | ′, ′,  | ′ ′ |     |
o n tw o d is ti n c ti n d ic e sa r e a s so c i at e d with lo n g - r an ge , in t er - f ie l d c con,vis gvis uvis x y t dy dx
| coupling.Appliedstimulus,whenappropriate,isgivenbyfunction |     |     |     |     |     |     |     |     |    |       |     |     |       |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ----- | --- | --- |
|                                                            |     |     |     |     |     |     |     | +   |     |  y−y | ′  |  ( | ′, ) | ′   |     |
s. Excitatory coupling takes positive values, while inhibitory c con,fAtn g fAtn u fAtn y t dy
| coupling | is negative. |     | The functional |     | topography | assumes | local |     |     |     |     |     |     |     |     |
| -------- | ------------ | --- | -------------- | --- | ---------- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |

e x c ita ti o n a n d la te ra l i n h i bi t io n ,a n d i t i s m o d e le d b y a d i f fe r en c e +  y−y ′   ′,  ′
|            |             |             |             |          |            |                 |                    |     | c con,wm |     | gwm | uwm y | t dy |     |     |
| ---------- | ----------- | ----------- | ----------- | -------- | ---------- | --------------- | ------------------ | --- | -------- | --- | --- | ----- | ---- | --- | --- |
| o f tw o G | a u s si an | s re su l t | i n g i n a | M e x ic | a n - h at | c o n n ec t iv | it y . T h e d o t |     |          |     |     |       |      |     |     |
˙
in u re pr e s e n t s t h e d e r iv a t i ve o f n e u ro n a l a c t iv it y uw i th r e s p e c t  ( )+η (y , t)+s ( ).
|     |     |     |     |     |     |     |     | +a  | con,nogo | ×g  | u   | t   |     | y   | (A.4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | ----- |
t o t im e t . D e t a il e d d e fi n it i o n so f e a c h c o u p li n g t e rm ar e in c l u d e d nogo nogo con con
in Appendices A.2–A.4, and the set of parameters used in the Similarly, the working memory field receives feedforward
simulationofthisDNFmodelarelistedinTablesA.2.1,A.3.1and excitatoryconnectionsfromthevisualandfeatureattentionfields;
A.4.1. inhibitory connections from the contrast field; and excitatory

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 231
ξ( , )
feedbackfromthegonode.Inaddition,weincludeasub-threshold Note that the variable z t takes random values from a
input swm (y) of activity bumps localized at the Go colors which normaldistributionwithzeromeanandu√nitstandarddeviation
simulateslearningduringthepre-taskinstructionstep, N ( 0 , 1 ) buthasitsstrengthscaledwith1 / dt.
τ e u˙ wm ( y  , t )=−uwm (y , t)+hwm spa S n i s m tw ila o r c d o e o f r in d i i t n i a o t n e s s, a t r h e e g s i p ve a n tia fo la r n th d e co v l i o su r a re l p fi r e e l s d e ( n j t = atio v n is s ) . w In h t i h c i h s
+ cwm  y−y ′ gwm  uwm  y ′, t  dy ′ case, the convolution cvis ∗ gvis (uvis ) and the noise η vis are two-
dimensionalfunctionssotheGaussianinteractionkernelandthe

+ cwm,vis  y−y ′ gvis  uvis  x ′, y ′, t  dy ′ dx ′ spatiallycorrelatednoisearedefinedby
+  cwm,fAtn  y−y ′ g fAtn  u fAtn ( y ′, t ) dy ′ c j  x−x ′, y−y ′=a j,E Exp  −  x 2 − σ j 2 , x E ′2 Exp  − ( y 2 − σ j 2 , y E ′)2 
+ ×g  go ( c u w g m o ( ,c t o ) n )  + y− η w y m ′ ( y g , co t n )  + u co s n w ( m y ( ′, y t ) ) .  dy ′+awm,go (A.5) + + a a j j , , I gl E ob x a p l  −  x 2 − σ j x 2 ,I ′2 Exp  −  y 2 − σ j y 2 ,I ′2 (A.11)
The go and nogo nodes are coupled by mutual inhibition. In
addition, feedforward excitation is projected from the working and
memoryfieldtothegonode,andfromthecontrastfieldtothenogo    x−x ′2
n τ e o u˙ d g e o ( r t e ) sp = ect × − iv u e g  g l n o y og . ( o t)  u + no h go g ( o t + ) a + go a × go, g w g m o  u go ( t )+a go,nogo η j (x , y , t) = ×Ex a p j,  no − ise  E y x − p y − ′2 2  σ ξ j 2 ,n  o x is ′ e , y ′, t  dx ′ dy ′. (A.12)
× gwm  uwm ( y ′, t ) dy ′+η go (t) (A.6) 2 σ j 2 ,noise
τ e u˙ nogo ( t ) = + −u a n n o o g g o o, ( g t o ) × + g h go no  g u o g + o ( t a ) n  og + o × a n g og n o o , g c o on  u nogo ( t ) { fi g e o l O d , n n in og t t e h o r e } a a c o r t e t i h o a e n s r s su a h m r a e n e s d d i , m to t p h h l e y av g t e h o e g a l p o n r b d o a d l n u c o o c g t n o ne n c o t d iv e i s ty w .T i h th en in th d e e i x rl j oca ∈ l

× g con  u con ( y ′, t ) dy ′+η nogo ( t ). (A.7) a ×g  u ( t ) (A.13)
j j j
betweenthegainfunctionandconstanta.Thenoisefunctionis
j
A.2. Localfieldinteractions definedby
All parameters associated with local interactions in the DNF
η
j
(
t
)=a
j,noise
×ξ(
t
).
(A.14)
modelabovearelistedinTableA.2.1.
TheGaussianinteractionkernelthatdeterminesthespreadof
activationinsideagivenfieldtoneighboringunits(seeparameters A.3. Longrange(inter-network)coupling
σ σ
j,E and j,I in Table A.2.1) with strengths determined by the
amplitudeparametersa j,E ,a j,I anda j,global isdefinedby Thecouplingbetweentwodistinctfieldsoftheneuralnetwork
c j  z−z ′ = a j,E Exp  − ( z 2 − σ j 2 , z E ′)2  −a j,I Exp  −  z 2 − σ j z 2 ,I ′2 i c i n s k, p j d ( u e ·) t fi f ∗ n ro e g m d j  b u fi y j e ( l · a d , t j G ) t a  h u e w s n s it i t a h h n e ke k c r o e n r n n e n l e e l ct a i s vi w ty e f l u l. n T c h ti u o s n , i i s f t f h ie e ld co k nv re o c lu e t iv io e n s
dim H e e n r s e io t n he ( x v + a ∈ r a ia j, b g S l l o ) e ba o l z . r t = he x fe o a r tu z re = (co y lo s r p ) a d n i s m e e i n th si e o r n th ( y e s ∈ p ( a A t F . i 8 a ) ) l , c k,j  z−z ′=a k,j Exp  − ( z 2 − σ k 2 z ,j ′)2  . (A.15)
whiletheindexj∈{sAtn , fAtn , con ,w m}correspondstotheneural
Inparticular,ifthecouplingisaprojectionofthevisualfield
fieldspatialattention,featureattention,contrastfieldorworking ( j=v
is
)
intoeitherofthefieldsspatialattention,featureattention,
memory,respectively.Thegainoutputfunctiong normalizesthe
( )
contrastorworkingmemory k ,thentheconvolutionisadouble-
fieldactivation,andisassumedtobethesigmoidal
integraloverthetwo-dimensionalset,S×F.TheGaussiankernel
1 depends, however, only on one variable (for example, x) so the
g
(
u
)=
1+Exp[−β u] (A.9) integration over the other variable ( y ) ultimately reduces to a
summationoftheoutputgainalongthesecondarydimensiony.
β
with threshold set to zero and steepness parameter . Conse-
If the coupling is a projection of the working memory (or
quently,activationlevelslowerthanthethresholdcontributerel-
contrast field) into the go (or nogo node), then the kernel of the
ativelylittletoneuralinteractions,whilepositiveactivationlevels
convolutionfunctionreducestoaconstant,
(higherthanthethreshold0)contributestronglytoneuralinter-
actions. c k,j =a k,j . (A.16)
Each neural network is subject to spatially correlated noise
η( z , t ) definedastheconvolutionbetweenaGaussiankerneland Inaddition,ifthecouplingisbetweenthegoandnogonodesthen
j
whitenoise ξ( z , t ) theconvolutionissimplytheproductc k,j × g j ( u j ( t )) and,again,
  c k,j =a k,j .
η j ( z , t )=  a j,noise Exp − ( 2 z σ − j 2 ,n z oi ′ s ) e 2 ξ z ′, t  dz ′. (A.10) lon T g a r b a l n e g A e . c 3 o .1 up s l u in m g m in ar t i h z e es D a N l F l p m a o r d am el. eter values associated with

232 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
TableA.2.1
Localfieldinteractions:parametervaluesusedinthesimulationoftheDNFmodel.SeealsoEqs.(A.1)–(A.5)and(A.8)–(A.14).
DifferencesinparametervaluesbetweenModel2(showninthetable)andModel1arehighlightedinredandshouldbe
readasfollows:Model1doesnotincludeany‘‘conditionofsatisfaction’’so,forit,lastcolumninthetableshouldbeignored.
Inaddition,inModel1,theamplitudeajofall-to-allcouplingforgoandnogonodesisfixedtoaGo =1andaNoGo =3(see
columns8and9inthetable).
TableA.3.1
Longrange(inter-network)coupling:parametervaluesusedinthesimulationoftheDNFmodel.
Forallexistingconnectionsjtokwhereitmakessense,thespreadofactivationtakesthevalue
σ k,j =5.SeealsoEqs.(A.6)–(A.7)and(A.15)–(A.16).DifferencesinparametervaluesbetweenModel
2(showninthetable)andModel1arehighlightedinredandshouldbereadasfollows:Model1
doesnotincludeany‘‘conditionofsatisfaction’’so,forit,lastrowandlastcolumninthetableshould
beignored.Inaddition,inModel1,thebi-directionalcouplingbetweenwmandconisacon,wm =
awm,con =−0.56andthebi-directionalcouplingbetweenconandNogoisacon,nogo =anogo,con =1.

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 233
TableA.4.1
Stimulusfunctions:parametervaluesusedinthesimulationoftheDNFmodel.SeealsoEqs.(A.1),(A.2),
(A.4),(A.5)and(A.17).
A.4. Stimulusfunctions oftheGo/Nogostimulussvis (x , y).Similarly,sub-thresholdactivity
|     |     |     |     |     |     |     |     | bumps in | the contrast | and working | memory | fields | are assumed |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | ----------- | ------ | ------ | ----------- |
toformduringtheinstructionstagewhenthesubjectlearnsthe
AllparametersassociatedwithstimuliintheDNFmodelappear
in Table A.4.1. Stimuli s j to field j are modeled by normalized GoandNogocolors,andagainpriortoapplicationoftheexternal
(x , y).Forexample,Load4requireslearningoftwoGo
| Gaussian | inputs | centered |     | at particular | position | z j,s in | the neural | stimulussvis |     |     |     |     |     |
| -------- | ------ | -------- | --- | ------------- | -------- | -------- | ---------- | ------------ | --- | --- | --- | --- | --- |
σ
field, and with spread parameter j,s and amplitude a j,s . In colorsandothertwoNogocolors.Therefore,duringthenumerical
particular, stimuli applied to the spatial attention, contrast and simulationtime,twosub-thresholdactivitybumpscenteredatthe
working memory fields induce local sub-threshold bump(s) of Gocolorsareplacedintheworkingmemoryfield,andtwosub-
,
activityintheabsenceoftheexternalstimulussvis (x y) thresholdactivitybumpscenteredattheNogocolorsareplacedin
thecontrastfield.
svis (x , y)=avis,s
|     |          |    |            | 2    |           |         |     |            |     |     |     |     |     |
| --- | -------- | --- | ---------- | ------ | ---------- | -------- | --- | ---------- | --- | --- | --- | --- | --- |
|     |          |     |  x−xvis,s |        | ( y−yvis,s | )2       |     |            |     |     |     |     |     |
|     | 1        |     |            |        |            |          |     | References |     |     |     |     |     |
|     | ×        | Exp | −          | Exp    | −          |          |     |            |     |     |     |     |     |
|     | πσ 2     |     | σ          | 2      |            | σ 2      |     |            |     |     |     |     |     |
|     | 2 v is,s |     | 2          | v is,s |            | 2 v is,s |     |            |     |     |     |     |     |
Amari,S.(1977).Dynamicsofpatternformationinlateral-inhibitiontypeneural
( )=a fields*.BiologicalCybernetics,27(2),77–87.http://doi.org/10.1007/BF00337259.
| s sAtn | x   | sAtn,s |     |     |     |     |     |     |     |     |     |     |     |
| ------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Anderson,J.R.(2012).Trackingproblemsolvingbymultivariatepatternanalysis
|     |     |     |  ( | )2  |    |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1 x−x sAtn,s a n d H i d d e n M a r k o v M o d e l a l g o r i th m s . N e u r o p s y c h ologia,50(4),487–498.http:
×√ Exp − // d oi .o r g / 1 0. 10 1 6 / j .n eu r o p s y c h o l o g ia. 2 0 1 1 . 0 7 . 0 2 5 .
|     | πσ  |     |     | σ 2 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2 sAtn,s 2 Atn,s Anderson,J.R.,Matessa,M.,&Lebiere,C.(1997).ACT-R:atheoryofhigherlevel
s
( )=a c o g n i ti o n a n d it s r e l a ti o n t o v i su a l a tt e n t ion.Human-ComputerInteraction,http:
s y con,s / / d o i. o r g /1 0 . 12 0 7 / s 1 5 3 2 7 0 5 1 h ci 1 2 0 4 _ 5 .
con
/2  ( )2  Aoki , R . , S a t o , H . , K a t u r a , T . , U t s u g i, K ., K o i z u m i , H . , M a t s u d a , R ., e t a l . ( 2 0 1 1 ) .
|     | 1   | lo  a d |     | y− y l |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
×√ − c on ,s R e l a t i o n s h i p o f n e g a t i v e m o o d w i t h p r e f r o n t a l c o r t e x a c t i v it y d u r i n g w o r k i n g
Exp σ m e m o r y t a s k s : A n o p t i c a l t o p o g r a p h y s t u d y . N e u r o s c i e n c e R e s e a r c h , 7 0 ( 2 ) ,
|     | 2 π σ | con,s |     | 2 2 | ,s  |     |     |     |     |     |     |     |     |
| --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
l = 1 c o n 1 8 9 – 1 9 6 . h t t p : / / d o i . o r g / 1 0 . 1 0 1 6 / j . n e u r e s . 2 0 1 1 . 0 2 . 0 1 1 .
( )=awm,s Aron , A . R . ,R o b b i n s, T . W ., & P o ld r a ck , R . A .( 20 1 4 ) . I n h ib it i o n a n d t h e r i g h t i n f e r io r
swm y
|     |     |         |     |        |     |     |     | fr o n t a l co  | r t e x : O n e d      | e ca d e o n . Tr e n d s in | C o g n i t iv e S c | i e nc e s, h | t t p: / / d o i. o r g / 1 0 . |
| --- | --- | ------- | --- | ------ | --- | --- | --- | ---------------- | ---------------------- | ---------------------------- | -------------------- | ------------- | ------------------------------- |
|     |     |         |     |       |    |     |     |                  |                        |                              |                      |               |                                 |
|     |     | lo ad/2 |     | ( y−yl | )2  |     |     | 1 0 1 6 / j .t i | c s .2 0 1 3 .1 2 .0 0 | 3 .                          |                      |               |                                 |
1  wm,s . Barb e y , A . K . , K o e n i gs , M ., & G rafman,J.(2013).Dorsolateralprefrontalcontributions
|     | ×√  |     | Exp | −   |     |     | (A.17) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
πσ 2 σ 2 t o h u m a n w o r k i n g m emory.Cortex,49(5),1195–1205.http://doi.org/10.1016/j.
|     | 2             | wm,s l=1 |          | w       | m,s         |           |          | c or t ex .2 | 0 1 2. 0 5 . 0 2 2 .  |                |             |           |             |
| --- | ------------- | -------- | -------- | ------- | ----------- | --------- | -------- | ------------ | --------------------- | -------------- | ----------- | --------- | ----------- |
|     |               |          |          |         |             |           |          | Bastian, A., | Riehle, A., Erlhagen, | W., & Schöner, | G. (1998).  | Prior     | information |
| The | sub-threshold |          | activity | bump in | the spatial | attention | field is |              |                       |                |             |           |             |
|     |               |          |          |         |             |           |          | preshapes    | the population        | representation | of movement | direction | in motor    |
assumedtoformduringthefixationstageandpriortoapplication cortex.NeuroReport,9(2),315–319.

234 S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235
Bastian,A.,Schöner,G.,&Riehle,A.(2003).Preshapingandcontinuousevolutionof Jirsa,V.K.,&Haken,H.(1997).Aderivationofamacroscopicfieldtheoryofthebrain
motorcorticalrepresentationsduringmovementpreparation.EuropeanJournal fromthequasi-microscopicneuraldynamics.PhysicaD:NonlinearPhenomena,
ofNeuroscience,18,2047–2058. 99(4),503–526.http://doi.org/10.1016/S0167-2789(96)00166-2.
Bennett,C.M.,&Miller,M.B.(2010).Howreliablearetheresultsfromfunctional Johnson, J. S., Hollingworth, A., & Luck, S. J. (2008). The role of attention in
magnetic resonance imaging? Annals of the New York Academy of Sciences, themaintenanceoffeaturebindingsinvisualshort-termmemory.Journalof
1191(1),133–155.http://doi.org/10.1111/j.1749-6632.2010.05446.x. ExperimentalPsychology:HumanPerceptionandPerformance,34,41–55.
Bicho,E.,&Schöner,G.(1997).Thedynamicapproachtoautonomousrobotics Johnson,J.S.,Spencer,J.P.,Luck,S.J.,&Schöner,G.(2009).Adynamicneuralfield
demonstrated on a low-level vehicle platform. Robotics and Autonomous modelofvisualworkingmemoryandchangedetection.PsychologicalScience,
Systems,21(1),23–35.http://doi.org/10.1016/S0921-8890(97)00004-3. 20(5),568–577.http://doi.org/10.1111/j.1467-9280.2009.02329.x.
Borst,J.P.,&Anderson,J.R.(2013).Usingmodel-basedfunctionalMRItolocate Johnson,J.S.,Spencer,J.P.,&Schöner,G.(2008).Movingtoahigherground:
workingmemoryupdatesanddeclarativememoryretrievalsinthefronto- thedynamicfieldtheoryandthedynamicsofvisualcognition.NewIdeasin
parietalnetwork.ProceedingsoftheNationalAcademyofSciencesoftheUnited Psychology,26,227–251.
StatesofAmerica,110(5),1628–1633.http://doi.org/10.1073/pnas.1221572110. Jonides,J.,Schumacher,E.H.,Smith,E.E.,Koeppe,R.A.,Awh,E.,Reuter-Lorenz,P.
Borst,J.P.,Nijboer,M.,Taatgen,N.A.,VanRijn,H.,&Anderson,J.R.(2015).Using A.,etal.(1998).Theroleofparietalcortexinverbalworkingmemory.Journal
data-drivenmodel-brainmappingstoconstrainformalmodelsofcognition. ofNeuroscience,18(13),5026–5034.
PLoSOne,10(3),http://doi.org/10.1371/journal.pone.0119673. Kaladjian, A., Jeanningros, R., Azorin, J.-M., Anton, J.-L., & Mazzola-Pomietto,
Boucher,L.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2007).Inhibitorycontrolinmind P. (2011). Impulsivity and neural correlates of response inhibition in
andbrain:aninteractiveracemodelofcountermandingsaccades.Psychological schizophrenia.PsychologicalMedicine,41(2),291–299.http://doi.org/10.1017/
Review,114(2),376–397.http://doi.org/10.1037/0033-295X.114.2.376. S0033291710000796.
Bressloff,N.W.(2001).Three-dimensionalpulsatileflowthroughasymmetrically Kelso,J.A.S.,Scholz,J.P.,&Schoner,G.(1988).Dynamicsgovernsswitchingamong
and symmetrically constricted vessels. In American society of mechanical patternsofcoordinationinbiologicalmovement.PhysicsLettersA,134(1),8–12.
engineers,bioengineeringdivision(publication)BED,Vol.50(pp.525–526). http://doi.org/10.1016/0375-9601(88)90537-3.
Buss,A.,&Spencer,J.P.2008.Theemergenceofrule-use:Adynamicneuralfield Klaes,C.,Schneegans,S.,Schöner,G.,&Gail,A.(2012).Sensorimotorlearningbiases
modeloftheDCCS.Retrievedfromcsjarchive.cogsci.rpi.edu/Proceedings/2008/ choicebehavior:alearningneuralfieldmodelfordecisionmaking.PLoSCompu-
pdfs/p463.pdf. tationalBiology,8(11),e1002774.http://doi.org/10.1371/journal.pcbi.1002774.
Buss,A.T.,&Spencer,J.P.(2014).Theemergentexecutive:Adynamicfieldtheory Kopecz, K., & Schöner, G. (1995). Saccadic motor planning by integrating
ofthedevelopmentofexecutivefunction.MonographsoftheSocietyforResearch visualinformationandpre-informationonneural,dynamicfields.Biological
inChildDevelopment,79(2),1–132. Cybernetics,73,49–69.
Buss,A.T.,Wifall,T.,Hazeltine,E.,&Spencer,J.P.(2013).Integratingthebehavioral Laing, C. R., & Chow, C. C. (2001). Stationary bumps in networks of spik-
andneuraldynamicsofresponseselectioninadual-taskparadigm:Adynamic ing neurons. Neural Computation, 13(7), 1473–1494. http://doi.org/10.1162/
neuralfieldmodelofDuxetal(2009).JournalofCognitiveNeuroscience,26, 089976601750264974.
334–351. Lipinski, J., Schneegans, S., Sandamirskaya, Y., Spencer, J. P., & Schöner, G.
Christopoulos, V., Bonaiuto, J., & Andersen, R. A. (2015). A biologically plau- (2012). A neurobehavioral model of flexible spatial language behaviors.
sible computational theory for value integration and action selection in JournalofExperimentalPsychology.Learning,Memory,Cognition,38,1490–1511.
decisions with competing alternatives. PLoS Computational Biology, 11(3), http://doi.org/10.1037/a0022643.
http://doi.org/10.1371/journal.pcbi.1004104. Lipinski,J.,Spencer,J.P.,&Samuelson,L.K.(2010).Correspondingdelay-dependent
Cisek,P.(2012).Makingdecisionsthroughadistributedconsensus.CurrentOpinion biasesinspatiallanguageandspatialmemory.PsychologicalResearch,74(3),
inNeurobiology,http://doi.org/10.1016/j.conb.2012.05.007. 337–351.http://doi.org/10.1007/s00426-009-0255-x.
Cohen,M.S.,&DuBois,R.M.(1999).Stability,repeatability,andtheexpression Logan,G.D.,Yamaguchi,M.,Schall,J.D.,&Palmeri,T.J.(2015).Inhibitorycontrol
of signal magnitude in functional magnetic resonance imaging. Journal of in mind and brain 2.0: Blocked-input models of saccadic countermanding.
MagneticResonanceImaging,10(1),33–40.http://doi.org/10.1002/(SICI)1522- PsychologicalReview,122(2),115–147.http://doi.org/10.1037/a0038893.
2586(199907)10:1<33::AID-JMRI5>3.0.CO;2-N. Mack,M.L.,Preston,A.R.,&Love,B.C.(2013).Decodingthebrain’salgorithm
Coombes,S.,&Owen,M.R.(2005).Bumps,breathers,andwavesinaneuralnetwork for categorization from its neural implementation. Current Biology, 23(20),
withspikefrequencyadaptation.PhysicalReviewLetters,94(14),http://doi.org/ 2023–2027.http://doi.org/10.1016/j.cub.2013.08.035.
10.1103/PhysRevLett.94.148102. Markounikau,V.,Igel,C.,&Jancke,D.2008.AmesoscopicmodelofVSDdynamics
Curtu,R.,&Ermentrout,B.(2004).Patternformationinanetworkofexcitatory observedinvisualcortexinducedbyflashedandmovingstimuli.frontiersin
andinhibitorycellswithadaptation.SIAMJournalonAppliedDynamicalSystems, humanneuroscience.conferenceabstract:10thinternationalconferenceon
3(3),191–231.http://doi.org/10.1137/030600503. cognitiveneuroscience.
Davis,T.,Love,B.C.B.,&Preston,A.R.A.(2012).Striatalandhippocampalentropy Markounikau, V., Igel, C., Grinvald, A., & Jancke, D. (2010). A dynamic neural
andrecognitionsignalsincategorylearning:simultaneousprocessesrevealed fieldmodelofmesoscopiccorticalactivitycapturedwithvoltage-sensitivedye
bymodel-basedfMRI.J.ExperimentalPsychology.Learning,Memory,Cognition, imaging.PLoSComputationalBiology,6(9),http://doi.org/10.1371/journal.pcbi.
38(4),821–839.http://doi.org/10.1037/a0027865. 1000919.
Deco,G.,Rolls,E.T.,&Horwitz,B.(2004).‘‘What’’and‘‘where’’invisualworking Markram,H.,Muller,E.,Ramaswamy,S.,Reimann,M.W.,Abdellah,M.,Sanchez,C.
memory:acomputationalneurodynamicalperspectiveforintegratingFMRI A.,etal.(2015).Reconstructionandsimulationofneocorticalmicrocircuitry.
and single-neuron data. Journal of Cognitive Neuroscience, 16(4), 683–701. Cell,163(2),456–492.http://doi.org/10.1016/j.cell.2015.09.029.
http://doi.org/10.1162/089892904323057380. McDowell,K.,Jeka,J.J.,Schöner,G.,&Hatfield,B.D.1998.Testingdynamicfield
Dux, P. E., Tombu, M. N., Harrison, S., Rogers, B. P., Tong, F., & Marois, R. theorywithdirectionalarmmovements.In28thannualmeetingofthesociety
(2009).Trainingimprovesmultitaskingperformancebyincreasingthespeed forneuroscience.LosAngeles,CA.
ofinformationprocessinginhumanprefrontalcortex.Neuron,63(1),127–138. McDowell, K., Jeka, J. J., Schöner, G., & Hatfield, B. D. (2002). Behavioral and
http://doi.org/10.1016/j.neuron.2009.06.005. electrocorticalevidenceofaninteractionbetweenprobabilityandtaskmetrics
Erika-Florence,M.,Leech,R.,&Hampshire,A.(2014).Afunctionalnetworkperspec- inmovementpreparation.ExperimentalBrainResearch,144,303–313.
tiveonresponseinhibitionandattentionalcontrol.NaturalCommunications, Mechelli, A., Humphreys, G. W., Mayall, K., Olson, A., & Price, C. J. (2000).
5(May),4073.http://doi.org/10.1038/ncomms5073. Differentialeffectsofwordlengthandvisualcontrastinthefusiformandlingual
Erlhagen,W.,Bastian,A.,Jancke,D.,Riehle,A.,&Schöner,G.(1999).Thedistribution gyriduringreading.ProceedingsofBiologicalSciences,267(1455),1909–1913.
of neuronal population activation (DPA) as a tool to study interaction and http://doi.org/10.1098/rspb.2000.1229.
integration in cortical representations. Journal of Neuroscience Methods, 94, Monterosso,J.R.,Aron,A.R.,Cordova,X.,Xu,J.,&London,E.D.(2005).Deficits
53–66. inresponseinhibitionassociatedwithchronicmethamphetamineabuse.Drug
Erlhagen,W.,&Schöner,G.(2002).Dynamicfieldtheoryofmovementpreparation. AlcoholDependence,79(2),273–277.http://doi.org/10.1016/j.drugalcdep.2005.
PsychologicalReview,109,545–572. 02.002.
Ermentrout, B. (1998). Neural networks as spatio-temporal pattern-forming Munk,M.H.J.,Linden,D.E.J.,Muckli,L.,Lanfermann,H.,Zanella,F.E.,Singer,W.,
systems.ReportsonProgressinPhysics,61(4),353–430.http://doi.org/10.1088/ etal.(2002).Distributedcorticalsystemsinvisualshort-termmemoryrevealed
0034-4885/61/4/002. byevent-relatedfunctionalmagneticresonanceimaging.CerebralCortex(New
Ermentrout,B.,&Kleinfeld,D.(2001).Travelingelectricalwavesincortex.Neuron, York,N.Y.:1991),12(8),866–876.http://doi.org/10.1093/cercor/12.8.866.
29,33–44. Palmeri, T. J., Schall, J. D., & Logan, G. D. (2015). Neurocognitive modeling
Garagnani, M., Wennekers, T., & Pulvermüller, F. (2008). A neuroanatomi- of perceptual decision making. In The oxford handbook of computational
cally grounded hebbian-learning model of attention-language interactions andmathematicalpsychology(pp.320–340).http://doi.org/10.1093/oxfordhb/
in the human brain. European Journal of Neuroscience, 27(2), 492–513. 9780199957996.013.15.
http://doi.org/10.1111/j.1460-9568.2008.06015.x. Perone, S., Simmering, V. R., & Spencer, J. P. (2011). Stronger neural dynamics
Gerstner,W.,Sprekeler,H.,&Deco,G.(2012).Theoryandsimulationinneuro- capture changes in infants’ visual working memory capacity over develop-
science.Science,338(6103),60–65.http://doi.org/10.1126/science.1227356. ment.DevelopmentalScience,14(6),1379–1392.http://doi.org/10.1111/j.1467-
Hampshire, A., & Sharp, D. J. (2015). Contrasting network and modular per- 7687.2011.01083.x.
spectivesoninhibitorycontrol.TrendsinCognitiveSciences,19(8),445–452. Perone,S.,&Spencer,J.P.(2013).Autonomyinaction:linkingtheactoflookingto
http://doi.org/10.1016/j.tics.2015.06.006. memoryformationininfancyviadynamicneuralfields.CognitiveScience,37(1),
Haxby,J.V,Petit,L.,Ungerleider,L.G.,&Courtney,S.M.(2000).Distinguishing 1–60.http://doi.org/10.1111/cogs.12010.
thefunctionalrolesofmultipleregionsindistributedneuralsystemsforvisual Perone, S., & Spencer, J. P. (2013a). Autonomous visual exploration creates
working memory. NeuroImage, 11(5 Pt 1), 380–391. http://doi.org/10.1006/ developmentalchangeinfamiliarityandnoveltyseekingbehaviors.Frontiers
nimg.2000.0592. inPsychology,4(September),648.http://doi.org/10.3389/fpsyg.2013.00648.

S.Wijeakumaretal./JournalofMathematicalPsychology76(2017)212–235 235
Perone,S.,&Spencer,J.P.(2013b).Autonomyinaction:Linkingtheactoflookingto Schutte,A.R.,&Spencer,J.P.(2002).GeneralizingthedynamicfieldtheoryoftheA-
memoryformationininfancyviadynamicneuralfields.CognitiveScience,37(1), not-Berrorbeyondinfancy:Three-year-olds’delay-andexperience-dependent
1–60.http://doi.org/10.1111/cogs.12010. locationmemorybiases.ChildDevelopment,73,377–404.
Perone, S., & Spencer, J. P. (2014). The co-development of looking dynamics Schutte, A. R., & Spencer, J. P. (2007). Planning discrete movements using
and discrimination performance. Developmental Psychology, 50(3), 837–852. a continuous system: Insights from a dynamic field theory of movement
http://doi.org/10.1037/a0034137. preparation.MotorControl,11,166–208.
Perone,S.,Spencer,J.P.,&Schöner,G.(2007).Adynamicfieldtheoryofvisual Schutte, A. R., Spencer, J. P., & Schöner, G. (2003). Testing the dynamic field
recognitionininfantlookingtasks.InD.S.McNamara,&J.G.Trafton(Eds.), theory:Workingmemoryforlocationsbecomesmorespatiallypreciseover
Proceedingsofthetwenty-ninthannualcognitivesciencesociety(pp.1391–1396). development.ChildDevelopment,74(5),1393–1417.
Nashville,TN:CognitiveScienceSociety. Simmering,V.R.,Peterson,C.,Darling,W.G.,&Spencer,J.P.(2008).Location
Pessoa,L.,Gutierrez,E.,Bandettini,P.,&Ungerleider,L.(2002).Neuralcorrelates memorybiasesrevealthechallengesofcoordinatingvisualandkinesthetic
of visual working memory: fMRI amplitude predicts task performance. referenceframes.ExperimentalBrainResearch,184,165–178.
Neuron,35(5),975–987.Retrievedfromhttp://www.ncbi.nlm.nih.gov/pubmed/ Simmering,V.R.,&Spencer,J.P.(2007).Carvingupspaceatimaginaryjoints:Can
12372290. peoplementallyimposespatialcategoryboundaries?.JournalofExperimental
Pessoa,L.,&Ungerleider,L.G.(2004).Neuralcorrelatesofchangedetectionand Psychology:HumanPerception&Performance,33,871–894.
changeblindnessinaworkingmemorytask—cerebcortex.CerebralCortex Smith,L.B.,Thelen,E.,Titzer,R.,&McLin,D.(1999).Knowinginthecontextof
(NewYork,N.Y.:1991),14(5),511–520.http://doi.org/10.1093/cercor/bhh013. acting:ThetaskdynamicsoftheA-not-Berror.PsychologicalRev.,106,235–260.
Pliszka,S.R.,Liotti,M.,&Woldorff,M.G.(2000).Inhibitorycontrolinchildren Smith,S.M.,Fox,P.T.,Miller,K.L.,Glahn,D.C.,Fox,P.M.,Mackay,C.E.,etal.
withattention-deficit/hyperactivitydisorder:Event-relatedpotentialsidentify (2009).Correspondenceofthebrain’sfunctionalarchitectureduringactivation
theprocessingcomponentandtimingofanimpairedright-frontalresponse- andrest.ProceedingsoftheNationalAcademyofSciencesoftheUnitedStatesof
inhibitionmechanism.BiologicalPsychiatry,48(3),238–246.http://doi.org/10. America,106(31),13040–13045.http://doi.org/10.1073/pnas.0905267106.
1016/S0006-3223(00)00890-8. Spencer,J.P.,&Schoner,G.(2003).Bridgingtherepresentationalgapinthedynamic
Rescorla,R.A.,&Wagner,A.R.1972.AtheoryofPavlovianconditioning:Variations systemsapproachtodevelopment.DevelopmentalScience,6(4),392–412.http:
intheeffectivenessofreinforcementandnonreinforcement.(InA.H.B). //doi.org/10.1111/1467-7687.00295.
Samuelson, L. K., Jenkins, G. W., & Spencer, J. P. (2015). Grounding cognitive- Thelen, E., Schöner, G., Scheier, C., & Smith, L. B. (2001). The dynamics of
levelprocessesinbehavior:Theviewfromdynamicsystemstheory.Topicsin embodiment:Afieldtheoryofinfantperseverativereaching.Behavioral&Brain
CognitiveScience,7(2),191–205.http://doi.org/10.1111/tops.12129. Science,24,1–86.
Sandamirskaya,Y.,&Schöner,G.2008.Dynamicfieldtheoryofsequentialaction: Turner, B. M., Forstmann, B. U., Love, B. C., Palmeri, T. J., & Van Maanen, L.
amodelanditsimplementationonanembodiedagent.In.Proceedingsofthe (2016).Approachestoanalysisinmodel-basedcognitiveneuroscience.Journal
internationalconferenceondevelopmentandlearning. ofMathematicalPsychology,http://doi.org/10.1016/j.jmp.2016.01.001.
Sandamirskaya,Y.,&Schöner,G.(2010).Anembodiedaccountofserialorder:how Turvey,MT,S.R.(1995).InM.D.SolsoR(Ed.),Thescienceofthemind:2001and
instabilitiesdrivesequencegeneration.NeuralNetworks:TheOfficialJournalof beyond,towardanecologicalphysicsandaphysicalpsychology.Oxford:Oxford
theInternationalNeuralNetworkSociety,23(10),1164–1179.http://doi.org/10. UniversityPress.
1016/j.neunet.2010.07.012. Usher,M.,&McClelland,J.L.(2001).Thetimecourseofperceptualchoice:the
Sandamirskaya,Y.,Zibner,S.K.U.,Schneegans,S.,&Schöner,G.(2013).Using leaky,competingaccumulatormodel.PsychologicalReview,http://doi.org/10.
dynamic field theory to extend the embodiment stance toward higher 1037/0033-295X.108.3.550.
cognition.NewIdeasinPsychology,31(3),322–339.http://doi.org/10.1016/j. Wiecki,T.V,&Frank,M.J.(2013).Acomputationalmodelofinhibitorycontrolin
newideapsych.2013.01.002. frontalcortexandbasalganglia.PsychologicalReview,120(2),329–355.http://
Schneegans, S., & Schöner, G. (2012). A neural mechanism for coordinate doi.org/10.1037/a0031542.
transformationpredictspre-saccadicremapping.BiologicalCybernetics,106(2), Wijeakumar,S.,Magnotta,V.A.,Buss,A.T.,Ambrose,J.P.,Wifall,T.A.,Hazeltine,
89–109.http://doi.org/10.1007/s00422-012-0484-8. E., et al. (2015). Response control networks are selectively modulated by
Schneegans,S.,Spencer,J.P.,Schöner,G.,Hwang,S.,&Hollingworth,A.(2014). attentiontorareeventsandmemoryloadregardlessoftheneedforinhibition.
Dynamic interactions between visual working memory and saccade target NeuroImage,120,331–344.http://doi.org/10.1016/j.neuroimage.2015.07.026.
selection.JournalofVision,14(11)(9),1–23.http://doi.org/10.1167/14.11.9. Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional
Schöner,G.,&Kelso,J.A.S.(1988).Dynamicpatterngenerationinbehavioraland dynamicsofcorticalandthalamicnervoustissue.Kybernetik,13,55–80.
neuralsystems.Science,239,1513–1520. Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism of time
Schöner,G.,Spencer,J.P.,&TheDFTResearchGroup,(2015).Dynamicthinking:a integrationinperceptualdecisions.JournalofNeuroscience,26(4),1314–1328.
primerondynamicfieldtheory.NewYork,NY:OxfordUniversityPress. http://doi.org/10.1523/JNEUROSCI.3733-05.2006.