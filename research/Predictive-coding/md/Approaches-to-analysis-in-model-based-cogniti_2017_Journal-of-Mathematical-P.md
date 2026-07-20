JournalofMathematicalPsychology76(2017)65–79
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Approaches to analysis in model-based cognitive neuroscience✩
BrandonM.Turnera,∗ ,BirteU.Forstmannb,BradleyC.Lovec,ThomasJ.Palmerid,
LeendertVanMaanenb
aDepartmentofPsychology,TheOhioStateUniversity,UnitedStates
bDepartmentofPsychology,UniversityofAmsterdam,Netherlands
cDepartmentofPsychology,UniversityCollege,London,UnitedKingdom
dDepartmentofPsychology,VanderbiltUniversity,UnitedStates
h i g h l i g h t s
• Wereviewcurrentapproachesforlinkingneuralandbehavioraldata.
• Wecompareandcontrastthesecurrentapproachesonavarietyoffactors.
• Weprovideaguidelineforselectingtheappropriateapproachinavarietyofcontexts.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Our understanding of cognition has been advanced by two traditionally non-overlapping and non-
Availableonline17February2016 interacting groups. Mathematical psychologists rely on behavioral data to evaluate formal models of
cognition,whereascognitiveneuroscientistsrelyonstatisticalmodelstounderstandpatternsofneural
Keywords:
activity,oftenwithoutanyattempttomakeaconnectiontothemechanismsupportingthecomputation.
Model-basedcognitiveneuroscience
Bothapproachessufferfromcriticallimitationsasadirectresultoftheirfocusondataatonelevelof
Linking
analysis(cf.Marr,1982),andtheselimitationshaveinspiredresearcherstoattempttocombineboth
Analysismethods
neural and behavioral measures in a cross-level integrative fashion. The importance of solving this
problemhasspawnedseveralentirelynewtheoreticalandstatisticalframeworksdevelopedbyboth
mathematicalpsychologistsandcognitiveneuroscientists.However,witheachnewapproachcomesa
particularsetoflimitationsandbenefits.Inthisarticle,wesurveyandcharacterizeseveralapproaches
forlinkingbrainandbehavioraldata.Weorganizetheseapproachesonthebasisofparticularcognitive
modelinggoals:(1)usingtheneuraldatatoconstrainabehavioralmodel,(2)usingthebehavioralmodel
topredictneuraldata,and(3)fittingbothneuralandbehavioraldatasimultaneously.Withineachgoal,
we highlight a few particularly successful approaches for accomplishing that goal, and discuss some
applications.Finally,weprovideaconceptualguidetochoosingamongvariousanalyticapproachesin
performingmodel-basedcognitiveneuroscience.
©2016ElsevierInc.Allrightsreserved.
1. Introduction theoriesbydevelopingformalmodelsofcognition. Themodelsof-
tenassumeasystemofcomputationsandmathematicalequations
Our understanding of cognition has been advanced by two intendedtocharacterizeaprocessthatmightactuallytakeplace
nearly non-overlapping and non-interacting groups. The first inthebrain.Toformallytesttheirtheory,mathematicalpsycholo-
group, mathematical psychologists, is strongly motived by the- gistsrelyontheirmodel’sabilitytofitbehavioraldata.Agoodfit
oretical accounts of cognitive processes, and instantiates these isthoughttoreflectanaccuratetheory,whereasabadfitwould
refute it (Roberts & Pashler, 2000). The second group, cognitive
neuroscientists,relyonstatisticalmodelstounderstandpatterns
of neural activity, often without any attempt to make a connec-
✩
This work was supported by grants NIH R01-EY021833, NSF Temporal
tiontothecomputationsthatmightunderliesomehypothesized
DynamicsofLearningCenterSMA-1041755,andNIHP30-EY08126.Thesecond
mechanism.Forexample,somestatisticalapproaches(e.g.,multi-
throughfifthauthorsaresimplylistedalphabetically.
∗ Correspondingauthor. variatepatternanalysis)explicitlyconditionontheneuraldatato
E-mailaddress:turner.826@gmail.com(B.M.Turner). determinewhichaspectsofthedataproducebetterpredictionsfor
http://dx.doi.org/10.1016/j.jmp.2016.01.001
0022-2496/©2016ElsevierInc.Allrightsreserved.

66 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
behavioraloutcomes.Suchananalysiscantelluswhichbrainre- speculate about how these methods can productively constrain
gionsarepredictiveofaparticularbehaviorandevenbyhowmuch, theory.Weclosewithadiscussionaboutadditionalconsiderations
buttheysaynothingaboutneitherhownorwhyparticularbrain in model-based cognitive neuroscience, and provide an outlook
| regionsproducesaidbehavior. |     |     |     |     |     |     |     | towardfuturedevelopment. |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- |
Althoughbothgroupsareconcernedwithexplainingbehavior,
| they tend        | to approach   |          | the challenge |               | from different |               | vantage   |                               |     |     |     |     |     |     |     |
| ---------------- | ------------- | -------- | ------------- | ------------- | -------------- | ------------- | --------- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                  |               |          |               |               |                |               |           | 2. Specificanalyticapproaches |     |     |     |     |     |     |     |
| points. Thinking |               | in terms | of            | Marr (1982)’s | levels         | of            | analysis, |                               |     |     |     |     |     |     |     |
| mathematical     | psychologists |          | tend          | to focus      | on the         | computational |           |                               |     |     |     |     |     |     |     |
Foreaseofcategorizationandsubsequentcomparison,wewill
| and algorithmic |     | levels, | whereas | cognitive | neuroscientists |     | focus |     |     |     |     |     |     |     |     |
| --------------- | --- | ------- | ------- | --------- | --------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
hypotheticallyassumethepresenceofneuraldata,denotedN,and
moreontheimplementationlevel.Althoughprogresscanbemade
behavioraldata,denotedB,whichmayormaynothavebeencol-
bymaintainingatightfocus,certainopportunitiesaremissed.As
|          |          |              |        |      |            |        |      | lected simultaneously. |     |     | The neural | data | N could | be  | neurophysio- |
| -------- | -------- | ------------ | ------ | ---- | ---------- | ------ | ---- | ---------------------- | --- | --- | ---------- | ---- | ------- | --- | ------------ |
| a result | of their | single-level | focus, | both | approaches | suffer | from |                        |     |     |            |      |         |     |              |
logicalrecordings,functionalmagneticresonanceimaging(fMRI),
| critical limitations |              | (Love,     | 2015).        | Without   | a cognitive     | model           | to       |                        |     |          |          |               |                |     |           |
| -------------------- | ------------ | ---------- | ------------- | --------- | --------------- | --------------- | -------- | ---------------------- | --- | -------- | -------- | ------------- | -------------- | --- | --------- |
|                      |              |            |               |           |                 |                 |          | electroencephalography |     |          | (EEG),   | or other      | physiological  |     | measures. |
| guide the            | inferential  | process,   | cognitive     |           | neuroscientists | are             | often    |                        |     |          |          |               |                |     |           |
|                      |              |            |               |           |                 |                 |          | The behavioral         |     | data B   | could be | response      | probabilities, |     | response  |
| (1) unable           | to interpret |            | their results | from      | a mechanistic   |                 | point of |                        |     |          |          |               |                |     |           |
|                      |              |            |               |           |                 |                 |          | times, confidence      |     | ratings, | or       | other typical | behavioral     |     | data col- |
| view, (2)            | unable       | to address | many          | phenomena |                 | when restricted |          |                        |     |          |          |               |                |     |           |
lectedinacognitiveexperiment.Cognitivemodelersareinterested
| to contrast | analyses, | and | (3) | unable | to bring | together | results |                   |     |     |            |     |           |     |              |
| ----------- | --------- | --- | --- | ------ | -------- | -------- | ------- | ----------------- | --- | --- | ---------- | --- | --------- | --- | ------------ |
|             |           |     |     |        |          |          |         | in characterizing |     | the | mechanisms | –   | specified | in  | mathematical |
fromdifferentparadigmsinacommontheoreticalframework.On
andcomputationalterms–thatleadtothebehaviorBobserved
theotherhand,thecognitivemodelsdevelopedbymathematical
inagivenexperimentalcondition.Commonly,thischaracteriza-
| psychologists | are | inherently |     | abstract, | and the | importance | of  |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------- | --- | --------- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionisderivedfromfittingacognitivemodeltobehavioraldata,
| physiology | and | brain | function | is often | unappreciated. |     | After |              |     |           |           |     |            |     |           |
| ---------- | --- | ----- | -------- | -------- | -------------- | --- | ----- | ------------ | --- | --------- | --------- | --- | ---------- | --- | --------- |
|            |     |       |          |          |                |     |       | interpreting | the | resulting | parameter |     | estimates, | and | comparing |
fittingamodeltodata,mathematicalpsychologistscandescribe
|                 |     |           |     |          |             |       |     | (qualitatively | or  | quantitatively) |     | the | observed | behavior | and the |
| --------------- | --- | --------- | --- | -------- | ----------- | ----- | --- | -------------- | --- | --------------- | --- | --- | -------- | -------- | ------- |
| an individual’s |     | behavior, | but | they can | say nothing | about | the |                |     |                 |     |     |          |          |         |
behavior’s neural basis. More importantly, neural data can behaviorpredictedbythemodel.Cognitiveneuroscientistsarein-
terestedinuncoveringtheneuralmechanismsthatleadtothebe-
provideinformationthatcanhelpdistinguishbetweencompeting
haviorBobservedinagivenexperimentalcondition.Commonly,
cognitivemodelsthatcannotbeuniquelyidentifiedbasedonfits
|     |     |     |     |     |     |     |     | this process | involves |     | a statistical | analysis |     | of neural | data with |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --- | ------------- | -------- | --- | --------- | --------- |
tobehavioraldataalone(Ditterich,2010;Mack,Preston,&Love,
|     |     |     |     |     |     |     |     | respect | to observed | behaviors |     | and experimental |     | manipulations. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | --------- | --- | ---------------- | --- | -------------- | --- |
2013;Purcell,Schall,Logan,&Palmeri,2012).
However,model-basedcognitiveneuroscientistsareinterestedin
| The many | limitations |     | of single-level |     | analyses | have | inspired |     |     |     |     |     |     |     |     |
| -------- | ----------- | --- | --------------- | --- | -------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
integratingneurophysiologicalinformationNandbehavioralout-
| researchers | to combine |     | neural | and behavioral |     | measures | in an |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | ------ | -------------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
comesBbywayofacognitivemodel.Thecentralassumptionof
| integrative | fashion.    | The     | importance |          | of solving      | the integration |     |                |     |         |             |          |     |      |               |
| ----------- | ----------- | ------- | ---------- | -------- | --------------- | --------------- | --- | -------------- | --- | ------- | ----------- | -------- | --- | ---- | ------------- |
|             |             |         |            |          |                 |                 |     | these analyses |     | is that | information | obtained |     | from | either source |
| problem     | has spawned | several |            | entirely | new statistical | modeling        |     |                |     |         |             |          |     |      |               |
approaches developed through collaborations between mathe- of data (N or B) can tell a similar story – albeit in different lan-
matical psychologists and cognitive neuroscientists, collectively guages–aboutsomeaspectofcognition,andtheintegrationofthe
forming a new field often referred to as model-based cogni- thesemeasuresassimilatesthedifferencesinlanguagesacrossdata
modalities.
| tive neuroscience |     | (e.g., | Boehm, | Van Maanen, | Forstmann, |     | & Van |                |     |           |     |                  |     |     |           |
| ----------------- | --- | ------ | ------ | ----------- | ---------- | --- | ----- | -------------- | --- | --------- | --- | ---------------- | --- | --- | --------- |
|                   |     |        |        |             |            |     |       | As model-based |     | cognitive |     | neuroscientists, |     | we  | have many |
Rijn,2014;Forstmann,Wagenmakers,Eichele,Brown,&Serences,
|             |       |      |         |       |          |       |          | choices | in deciding | which | story | we would | like | to tell, | and these |
| ----------- | ----- | ---- | ------- | ----- | -------- | ----- | -------- | ------- | ----------- | ----- | ----- | -------- | ---- | -------- | --------- |
| 2011; Love, | 2015; | Mack | et al., | 2013; | Palmeri, | 2014; | Palmeri, |         |             |       |       |          |      |          |           |
choicesdependonourresearchgoals.Inpractice,thereseemstobe
Schall,&Logan,2015;Turneretal.,2013b;Turner,VanMaanen,
atleastthreegeneralcategoriesofapproachesintheemergingfield
&Forstmann,2015b;vanMaanenetal.,2011).Werefertotheseas
‘‘approaches’’,becausetheyaregeneralstrategiesforintegrating ofmodel-basedcognitiveneuroscience.Thesethreecategoriesare
neuralandbehavioralmeasuresviacognitivemodels,andarenei- illustrated in the rows of Fig. 1. The first set of approaches uses
therrestrictedtoanyparticularkindofneuralorbehavioralmea- neural data as auxiliary information that guides or constrains a
sure,noranyparticularcognitivemodel.However,witheachnew behavioral model. There are several ways in which the neural
approachcomesauniquesetoflimitationsandbenefits.Theap- data can constrain modeling choices, and we will discuss three
proachesthathaveemergedintherecentyearsfillanentirespec- such approaches in the subsequent sections. The second set of
approachesusesabehavioralmodelasawaytointerpretorpredict
trumofinformationflowbetweenneuralandbehaviorallevelsof
neuraldata.Behavioralmodelsassumeasetofmechanismsthat
| analysis, | and deciding | between |     | them | can be difficult. | Given | the |     |     |     |     |     |     |     |     |
| --------- | ------------ | ------- | --- | ---- | ----------------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
theoreticallymimicacognitiveprocessofinterest,makingtheman
overwhelmingdemandfortheseintegrativestrategies,webelieve
thatanarticlesurveyingthedifferenttypesofanalyticapproaches interestingwaytoimposetheoryindataanalyses.Moreover,while
couldbeaninvaluableguideforanywould-bemodel-basedcog- competing cognitive models might predict the same or similar
nitiveneuroscientist. patterns of behavioral data B, they might differ considerably
Here we survey and characterize the many approaches for in what they predict about neural data N, creating a powerful
linking brain and behavioral data. We organize these different approach to model selection. We are faced with many choices
approaches into three general categories: (1) using the neural in using these model mechanisms to guide our search for the
data to constrain a behavioral model, (2) using the behavioral interestingneuralsignatures.Inthesectionsthatfollow,wewill
discusstwosuchapproachesforaccomplishingthisgoal.Thethird
| model to | predict | neural | data, and | (3) | modeling | both neural | and |     |     |     |     |     |     |     |     |
| -------- | ------- | ------ | --------- | --- | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
setofapproachesbuildsasinglemodelthatjointlyaccountsfor
behavioraldatasimultaneously.Foreachspecificapproachwithin
eachcategory,wehighlightafewparticularlysuccessfulexamples, the random variation present in both the neural and behavioral
and discuss some applications. In an attempt to draw a detailed data. With the proper model in place, one can simultaneously
comparison between the approaches, we then organize each of achieve constraint on the behavioral model while retaining the
the approaches according to a variety of factors: the number abilitytointerprettheneuraldata.Inthesectionsthatfollow,we
of processing steps, the commitment to a particular theory, the willdiscusstwoapproachesdesignedtoaccomplishthisgoal.We
type of information flow, the difficulty of implementation, and do not necessarily think this is a comprehensive list; in fact, we
the type of exploration. In short, we discuss the ways in which suspectthatthereisroomforfurtherdevelopment,andpossibly
current approaches bind data at multiple levels of analysis, and thecreationofentirelynewanalyticapproaches.

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 67
Fig.1. Anillustrationofseveralapproachesusedforlinkingneuralandbehavioraldata,organizedbyspecificmodelinggoals.Nrepresentstheneuraldata,Brepresents
thebehavioraldata,N∗representssimulatedinternalmodelstates,andθ,δ,andΩrepresentmodelparameters.Whenanapproachisprocedural,progressionthrough
processingstagesisrepresentedbyarrowsofdecreasingdarkness(e.g.,theLatentInputApproach).Dashedlinesindicateconceptualconstraints(e.g.,theTheoretical
Approach),whereassolidlinesindicatestatisticalconstraints.
Fig.1representsthespecificapproachesasgraphicaldiagrams representthingslikediscriminabilityinthesignaldetectiontheory
where observable measures (i.e., data) are depicted as shaded model (Green & Swets, 1966), or the drift rate in the ‘‘diffusion
square nodes, latent model parameters are depicted as empty decision model’’1 (Forstmann et al., 2016; Ratcliff, 1978). When
circles,andarrowsdepictdependencies.Twooftheseapproaches a bimodal transformation is required, we generically denote the
θ
(i.e., Two-stage and Latent Input) require several processing parametersas (e.g.,theIntegrativeApproachinthebottom-right
stages, and we have represented the dependency structure of panelofFig.1).Forexample,intheACT-Rframework(Anderson,
θ
these stages as increasingly lighter shades of gray. Most of 2007), the set of parameters represents a sequence of module
these approaches require a transformation from the data space activations,andtheirvalueshavebimodaleffectsintheprediction
to a (latent) parameter space, and this transformation can be of both neural and behavioral measures. Some approaches in
unimodal(i.e.,concerningonlybehaviordataBorneuraldataN) our set require a simulation process where the parameters are
or bimodal (i.e., concerning both B and N simultaneously). The used to generate synthetic data, and we will denote these data
∗
parameterscandefineamechanisticmodel,likethosecommonly with an asterisk (e.g., N denotes predicted neural data in the
used by cognitive modelers, or they can define a statistical Latent Input Approach). Other approaches assume a secondary
model, like those commonly used by cognitive neuroscientists. projectionfromasetofseveralparameterspacestoagroup-level
When an unimodal transformation is required, we denote the parameter space, such as in hierarchical modeling. We denote
δ
parameters of the neural model which predict N as , and the
θ
parameters of the behavioral model which predict B as . The
δ
neural model parameters might be slopes or intercept terms
1 In this article, we refer to this model as the ‘‘diffusion decision model’’
fromagenerallinearmodel,orsomethingmoresophisticatedlike
followingForstmann,Ratcliff,andWagenmakers(2016).Thissamemodelhasbeen
thoseusedintopographiclatentsourceanalysis(Gershman,Blei,
calledothernamessuchasthe‘‘thediffusionmodel’’,the‘‘driftdiffusionmodel’’,
θ
Pereira, & Norman, 2011). The behavioral model parameters andthe‘‘Wienerdiffusionmodel’’.

68 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
these higher-level parameters as Ω (e.g., the Joint Modeling aspectsofthedecisionrule,facilitatingfurthersystematicexplo-
Approach inthe bottom-left panel ofFig. 1). As an example, the rationsofrepresentation,learning,andselectiveinfluence(i.e.,by
jointmodelingframework(Turneretal.,2013b)usesahierarchical aprocessreferredtoas‘‘lesioning’’).
(Bayesian)structureforbridgingtheconnectionbetweenneural Asanotherexample,considertheLeakyCompetingAccumula-
and behavioral measures. With these general assumptions and tor (LCA; Usher & McClelland, 2001) model. The LCA model was
notation in place, we can discuss how these various approaches proposedasaneurallyplausiblemodelforchoiceresponsetime
achievetheirintendedanalyticgoal. inak-alternativetask.Themodelpossessesmechanismsthatex-
tend other diffusion-type models (e.g., Ratcliff, 1978) by includ-
ingleakageandcompetitionbymeansoflateralinhibition.These
2.1. Neuraldataconstrainbehavioralmodel
additionalmechanismshaveproveneffectiveinexplaininghow,
forexample,timesensitivestimulusinformationcangivewayto
We begin our discussion with approaches that constrain a
differencesinindividualsubjectperformance.Forexample,Usher
behavioral model with neural data. In this endeavor, the neural
andMcClelland(2001)andTsetsos,Usher,andMcClelland(2011)
data are considered important, but only in the sense that they
haveshowntheeffectsofprimacyandrecencyforsomesubjects
inform the mechanisms in the behavioral model. We have
inatime-varyingstimulusinformationparadigm.Inthesemulti-
identifiedthreespecificapproaches(i.e.,seeFig.1):theTheoretical
alternativechoiceexperiments,oneresponseoptionmayreceive
Approach, the Two-stage Behavioral Approach, and the Direct
the strongest ‘‘input’’ (e.g., the brightness level) for the first 500
InputApproach.Wenowdiscusseachoftheseapproachesinturn.
ms,butthenthestimulitransitionsuchthatadifferentresponse
optionreceivesthestrongestinputrelativetothefirst.Inbothof
2.1.1. Theoreticalapproach
thesestudies,differentparameterizationsoftheLCAmodelwere
IntheTheoreticalApproach,psychologicaltheoriesaredevel- usedtodemonstratehowprimacyeffectscouldbeappreciatedby
opedonthebasisofconsiderationsfrombothneuroscienceand havingalargevalueforlateralinhibitionrelativetothestrengthof
behavioral data. The top left panel of Fig. 1 illustrates the Theo- theinput(i.e.,thedriftrate),andrecencyeffectscouldbecaptured
retical Approach as statistically independent models of the neu- throughalargeleakagetermrelativetotheinput(Tsetsosetal.,
ralandbehavioraldatabecausethelinkbetweenthesemeasures 2011;Usher&McClelland,2001).
is established only through the researcher themselves (i.e., rep- As a specific example of how the neurosciences have guided
resented by the dashed arrow). In this approach, the dominant theassumptionsintheLCAmodel,itiswellknownthatthefiring
procedure uses neural measures to inspire the development of rateofindividualneuronscanneverbenegative.However,these
psychologicalmodels.First,theresearcherobservesparticularas- firing rates can be attenuated by way of inhibition—a process
pects of brain function, such as information about the structure carried out by other neurons in the system. To instantiate these
(e.g., individual neurons or densely connected brain regions) or neuronal dynamics, the full LCA model enforces a constraint
function(e.g.,dorsalandventralpathwaysofvisualstimuluspro- such that if the degree of evidence for any choice alternative
cessing)ofthebrain.Next,theresearcherdevelopsamodelofbe- becomes negative, the degree of evidence for that accumulator
haviorthat,atitscore,abidesbytheseneuralobservations.With shouldberesettozero(Usher&McClelland,2001).Thefloor-on-
aninitialmodelstructureimposedbyN,theresearcherisnowable activation constraint was later found to be critical in capturing
toevaluatetherelativemeritsofnestedtheoreticalassumptions, patternsofindividualdifferencesinmulti-alternativechoicethat
andmakeincrementaladjustmentsinthemodeltoprovidebetter could not be captured by other diffusion-type models (Tsetsos
fitstobehavioraldataB.Unlikeotherapproachesdiscussedinthis etal.,2011).Itisworthnotingthatotherneurologicalconstraints
article,theTheoreticalApproachmaydrawinspirationfromphys- allow the LCA model to provide a unique characterization of
iologicaloranatomicalobservations,butthereisnomathematical behavioraldatathatwouldnototherwiseberealized;specifically,
orstatisticallinkbetweentheneuraldataN andeitherthemodel the role of lateral inhibition relative to leakage in the model
architectureorthemodelparametersthatpredictthebehavioral playsaninterestingroleincharacterizingsubject-specificpatterns
dataB. in behavioral data (Bogacz, Brown, Moehlis, Holmes, & Cohen,
Although the absence of an explicit link between neural and 2006; Bogacz, Usher, Zhang, & McClelland, 2007; Gao, Tortell,
behavioral data may seem craven, the Theoretical Approach has & McClelland, 2011; Ossmy et al., 2013; Purcell et al., 2012;
proventobeapowerfulframeworkformotivatingpsychological Teodorescu & Usher, 2013; Tsetsos, Gao, McClelland, & Usher,
theory.PerhapsthemostprominentexampleofaTheoreticalAp- 2012; Tsetsos et al., 2011; Turner & Sederberg, 2014; Turner,
proach is the enormous class of neural network models. Neural Sederberg,&McClelland,forthcoming;vanRavenzwaaij,vander
networkmodelshavealonghistory,withoneclassicexamplebe- Maas,&Wagenmakers,2012).
ingRosenblatt’sPerceptronmachine(Rosenblatt,1961).Inthede- Given the highly subjective nature of the neural constraints
velopmentofthePerceptron,Rosenblattmadechoicesinhismodel imposed on a behavioral model, it should not be surprising
thatreflectedoperationsobservedinindividualneurons,suchas that a great deal of controversy surrounds some applications of
thatthefiringofindividualneuronsshouldbediscrete(motivated the Theoretical Approach. While neural network modelers have
bytheMcCullogh–Pittsneuron;McCullogh&Pitts,1943).Although undoubtedlyderivedinspirationfromthebraininbuildingtheir
theseoriginalneuralnetworkmodelswereheavilycriticized(Min- models, the mechanistic implementation of these inspirations is
sky&Papert,1969),pioneeringworkallowingforcontinuousac- ofteninterpretedasastrongcommitment,whichopensthegates
tivationsinneuron-likeunits(Anderson,1977;Grossberg,1978; for scrutiny about plausibility and falsifiability (Massaro, 1988;
McClelland&Rumelhart,1981;Rumelhart,1977;Rumelhart&Mc- Minsky&Papert,1969;Roberts&Pashler,2000).Furthermore,in
Clelland, 1982) evolved neural network models into more com- some cases these additional neural mechanisms do not provide
plex and successful theoretical approaches such as the parallel anyadvantageintermsofquantitativefitstatisticstobehavioral
distributedprocessing(PDP;McClelland&Rumelhart,1986)mod- data over their simpler counterparts (e.g., see Ratcliff and Smith
els.Superficially,thesemodelsallowforthepresenceofindividual (2004),butalsoseeTeodorescuandUsher(2013)andTurneretal.
nodesembeddedwithinlayersofanetwork,andthesenodesare (forthcoming)foradifferentperspective).Insomecases,thereare
massivelyinterconnectedacrosslayers,resemblingneuralstruc- alsoconcernscenteredonthelevelofexplanationthatthemodel
turesinthebrain.Throughaprocessknownasback-propagation, provides(cf.Marr,1982).Ontheonehand,thestudyofindividual
PDPmodelscanbetrainedonbehavioraldatatolearnimportant neurons constitutes an exploration of Marr’s implementation

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 69
level of analysis (Broadbent, 1985; Kemp & Tenenbaum, 2008; very few efforts have been devoted to systematically studying
Pinker & Prince, 1988; Smolensky, 1988). On the other, the themodel’spredictionsforchoiceresponsetimedata.However,
| development | of  | a cognitive | model | involves |     | meandering | through |                      |     |      |      |           |     |         |           |
| ----------- | --- | ----------- | ----- | -------- | --- | ---------- | ------- | -------------------- | --- | ---- | ---- | --------- | --- | ------- | --------- |
|             |     |             |       |          |     |            |         | a few approximations |     | have | been | developed | for | fitting | purposes, |
thecomputationallevel—Marr’shighestlevelofanalysis(Shiffrin andtheseapproximationsbehavesimilarlytopopularmodelsin
&Nobel,1997).Towhatextentshouldtheimplementationlevel cognitivesciencesuchastheLCAmodel(Bogaczetal.,2006;Roxin
be reflected or imposed on the computational level (e.g., Frank, &Ledberg,2008;Wong&Wang,2006).
| 2015; Love, | 2015;   | Teller, | 1984)?  | For | example,   | if we believe | that       |                            |     |     |     |     |     |     |     |
| ----------- | ------- | ------- | ------- | --- | ---------- | ------------- | ---------- | -------------------------- | --- | --- | --- | --- | --- | --- | --- |
| individual  | neurons | have    | a floor | on  | activation | or are        | inherently |                            |     |     |     |     |     |     |     |
|             |         |         |         |     |            |               |            | 2.1.3. Directinputapproach |     |     |     |     |     |     |     |
‘‘leaky’’(i.e.,meaningtheyloseinformationovertime),shouldthis
restriction be imposed on the dynamics of racing accumulators The Two-stage Behavioral Approach represents one way in
inacognitivemodel(Zandbelt,Purcell,Palmeri,Logan,&Schall, whichtheneuraldatacanguidethebehavioralmodelthroughneu-
2015)?Theseaccumulatorsareintendedtoreflecttheamountof ralmodelparameters,butitiseasytoimagineotherapproaches
sensoryevidenceforeachalternative—evidencethatisapparently thataremoredirect.Forexample,ratherthantranslatingtheneu-
observedinmanybrainareas(includingthelateralintraparietal raldataN totheneuralmodelparameters δ ,andthenusing δ to
|                |     |             |         |     |        |                  |     | constrain | the behavioral |         | model    | parameters | θ ,      | we could | instead    |
| -------------- | --- | ----------- | ------- | --- | ------ | ---------------- | --- | --------- | -------------- | ------- | -------- | ---------- | -------- | -------- | ---------- |
| area, superior |     | colliculus, | frontal | eye | field, | and dorsolateral |     |           |                |         |          |            |          |          |            |
|                |     |             |         |     |        |                  |     | use the   | neural         | data to | directly | replace    | dynamics | of       | the behav- |
prefrontalcortex;Hanes&Schall,1996;Hanksetal.,2015;Horwitz
& Newsome, 1999, 2001; Kim & Shadlen, 1999; Purcell et al., ioralmodel.Thisalternativeapproachisnicelyillustratedbythe
2010,2012;Shadlen&Newsome,1996,2001),andsoitbegsthe Vanderbilt group (e.g., Palmeri et al., 2015; Purcell et al., 2010,
question:Which–ifany–levelsofdecisionmakingmodelsshould 2012).Theyexaminedperceptualdecisionmakingwithinthese-
reflect the function of individual neurons? If the accumulators quentialsamplingmodelarchitectureassumedbymodelslikethe
are to reflect the behavior of individual neurons, how might DDM(DDM;Ratcliff,1978),andtheLCAmodel(Usher&McClel-
this connection be formally established (Smith, 2010; Smith & land,2001),amongothers.Theyspecificallytestedthehypothe-
McKenzie, 2011)? Questions like this have been considered by sis that different types of neurons in the frontal eye field (FEF)
many other scientists (e.g., Broadbent, 1985; Frank, 2015; Love, carryoutdifferentcomputationsspecifiedinaccumulatormodels,
| 2015; Marr, | 1982; | Schall, | 2004; | Teller, | 1984), | and the | next two |        |                          |     |     |         |        |        |           |
| ----------- | ----- | ------- | ----- | ------- | ------ | ------- | -------- | ------ | ------------------------ | --- | --- | ------- | ------ | ------ | --------- |
|             |       |         |       |         |        |         |          | namely | that visually-responsive |     |     | neurons | in FEF | encode | the drift |
sections discuss two different ideas about how this connection ratedrivingthedecisionprocessandthatmovement-relatedneu-
shouldbemade. ronsinFEFinstantiatetheaccumulationprocessitself.Totestthis
|     |     |     |     |     |     |     |     | linking proposition |     | most | directly | (cf. Schall, | 2004; | Teller, | 1984), |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ---- | -------- | ------------ | ----- | ------- | ------ |
2.1.2. Two-stagebehavioralapproach theyreplacedtheparameterizedmechanismsthoughttobeem-
The first formal linking approach uses neurophysiology to bodied by the visually-responsive neurons, namely the time for
replace parameters of a behavioral model. For example, consider perceptual processing and the drift rate, with the neurophysi-
| a model                          | that explains |      | some  | neural        | data N                 | with parameters | δ     | ,           |               |              |                     |     |               |          |        |
| -------------------------------- | ------------- | ---- | ----- | ------------- | ---------------------- | --------------- | ----- | ----------- | ------------- | ------------ | ------------------- | --- | ------------- | -------- | ------ |
|                                  |               |      |       |               |                        |                 |       | ological    | data recorded | from         | visually-responsive |     |               | neurons. | Rather |
| andbehavioraldataBwithparameters |               |      |       |               | θ .Theneuralparameters |                 |       |             |               |              |                     |     |               |          |        |
|                                  |               |      |       |               |                        |                 |       | than having | abstract      | mathematical |                     | and | computational |          | compo- |
| δ could                          | be divided    | into | a set | of parameters |                        | characterizing  | a key |             |               |              |                     |     |               |          |        |
nentsspecifiedbyfreeparametersdrivethedecisionprocess,the
|     | δ   |     |     |     |     |     | δ   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
neural signal 1 , and a set of nuisance parameters 2 so that neural data (N) drove the decision process directly. To do this,
| δ = {δ | ,δ }. |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1 2 Now suppose the behavioral model parameters theneuraldatawereusedtodirectlyreplacecomponentsofthe
could be divided into a set of parameters that are reflective of modelthatwouldotherwisehavebeenlatent,andwouldneedto
|     |     | θ   |     |     |     | θ   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thebehavioralsignal ,andasetofparameters thatarenot. be estimated from behavioral data. The only remaining free pa-
|     |     | 1   |     |     |     | 2   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ThestructureoftheTwo-stageBehavioralApproachistosimply rameterswerethosethatdefinedthedecisionmakingarchitecture
θ
| replacethesetofparameters |     |     |     | withtheparametersoftheneural |     |     |     |     |     |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
δ 1 (i.e., race, feedforward, lateral, or gated accumulation), and that
signal .Werefertothisapproachasthe‘‘Two-stageBehavioral’’
1 definedspeed-accuracytradeoffs(i.e.,thresholdofaccumulation).
| approach | because | the connection |     | involves |     | two stages, | and that |     |     |     |     |     |     |     |     |
| -------- | ------- | -------------- | --- | -------- | --- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Whenconstrainedbyneuralinputs,theyobservedthatonlysome
behavioralmodelparametersarereplacedbyneuralparameters.
ofthevariousdecisionmakingarchitecturescouldfitthefullset
| This approach |     | makes a | strong | commitment |     | to how the | neural |     |     |     |     |     |     |     |     |
| ------------- | --- | ------- | ------ | ---------- | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
ofbehavioraldata(correctanderrorresponsetimedistributions
signalisrepresentedintheabstractmechanismsassumedbythe
|     |     |     |     |     |     |     |     | and response | probabilities). |     | They | were | then able | to  | distinguish |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --------------- | --- | ---- | ---- | --------- | --- | ----------- |
behavioralmodel,andasaresult,itisastrongerinstantiationof further between models based on how well the predicted accu-
neurophysiologythantheTheoreticalApproachdiscussedabove.
mulatormodeldynamicsmatchedtheobservedneuraldynamics
| The Two-stage |     | Behavioral |     | Approach | is  | nicely illustrated | by  |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------- | --- | -------- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inmovement-relatedneurons,theneuronstheyhypothesizedto
| the work | of Wang | and | colleagues | (Wong | &   | Wang, 2006), | who |     |     |     |     |     |     |     |     |
| -------- | ------- | --- | ---------- | ----- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
carryoutanaccumulationofevidence(seeLatentInputApproach
developedaspikingneuralnetworkmodelofperceptualdecision
below).
| making. | This model | aims | to  | account | for | the same | kinds of |     |     |     |     |     |     |     |     |
| ------- | ---------- | ---- | --- | ------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
AlthoughtheDirectInputApproachiscommonlyusedtofeed
behaviorsastheDDMandtheLCAmodel,butisfarlessabstract,
|                |     |              |     |         |          |                |     | neural data | into | a cognitive | model, | one | could | potentially | invert |
| -------------- | --- | ------------ | --- | ------- | -------- | -------------- | --- | ----------- | ---- | ----------- | ------ | --- | ----- | ----------- | ------ |
| with thousands |     | of simulated |     | spiking | neurons, | dense patterns | of  |             |      |             |        |     |       |             |        |
thedirectionofinfluenceinFig.1toanalyzetheneuraldataasa
excitatoryandinhibitoryconnections,poolsofneuronsassociated
with a single response, and the dynamics of individual neurons functionofsomebehavioralvariable,suchasaccuracy(e.g.,Eichele
defined by several differential equations. While the model has et al., 2008) or response time (e.g., Hanes & Schall, 1996;
Weissman,Roberts,Visscher,&Woldorff,2006).Oncetheneural
| dozens of | potentially | free | parameters, |     | most | of them are | defined |     |     |     |     |     |     |     |     |
| --------- | ----------- | ---- | ----------- | --- | ---- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
datahavebeensortedasafunctionofthelevelsofthebehavioral
| directly | by neural | data. | For | example, | the | time constants | of  |          |     |               |     |                  |     |           |      |
| -------- | --------- | ----- | --- | -------- | --- | -------------- | --- | -------- | --- | ------------- | --- | ---------------- | --- | --------- | ---- |
|          |           |       |     |          |     |                |     | outcome, | one | might analyze |     | the distribution |     | of neural | data |
integrationofdifferentinhibitoryandexcitatoryreceptortypesare
betweentheselevels(Woodman,Kang,Thompson,&Schall,2008).
baseddirectlyonphysiologicalmeasures.Whilelow-levelspiking
|     |     |     |     |     |     |     |     | Such a | procedure | has been | the | dominant | analytic | approach | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --------- | -------- | --- | -------- | -------- | -------- | --- |
neuralnetworkmodelsofthissortcapturewellmanyofthedetails
|     |     |     |     |     |     |     |     | neuroscience | since | its inception, |     | but is | not model-based, |     | and so |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | -------------- | --- | ------ | ---------------- | --- | ------ |
ofneuronsandneuralcircuitsandprovidereasonablefirst-order
predictions of behavioral data, they are difficult to simulate and wewillnotconsiderithere.However,themodel-basedanalogue
quantitative fits to behavioral data are simply impossible using of this analysis would be to use the model’s machinery to drive
evenstate-of-the-artcomputerhardware(seeUmakantha,Purcell, theanalysisofneuraldata.WerefertothisapproachastheLatent
&Palmeri,2016).Indeed,asaresultofthisadditionalcomplexity, InputApproach,andwilldiscussitfurtherinthenextsection.

70 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
2.2. Behavioralmodelpredictsneuraldata data through the general linear model. However, the estimates
|     |     |     |     |     |     |     |     | v and ϵ | are not | parameters; | instead, |     | they reflect | the | model’s |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ----------- | -------- | --- | ------------ | --- | ------- |
Anothersetofanalyticapproachesinvolvessearchingthebrain internal state for value and prediction error, respectively. This
| for areas | that | support | mechanisms | assumed |     | in the | behavioral |     |     |     |     |     |     |     |     |
| --------- | ---- | ------- | ---------- | ------- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
distinctionisimportantbecauseitseparatesthisanalyticapproach
model. Such a procedure allows one to interpret neural data fromotherpossibleTwo-stageapproaches,suchasinvanMaanen
through mechanisms in the model, which can potentially be etal.(2011),whichwediscussbelow.
more informative than behavioral data alone. We consider two Asthepreviousexamplemakesclear,LatentInputApproaches
approachesforaccomplishingthisgoal:theLatentInputandthe canidentifycandidateneuralsubstratesfortheoreticalconcepts,
Two-stageNeuralApproaches. suchaspredictionerror,thatarenotdirectlyobservablebutcanbe
definedwithinacognitivemodel.Enteringlatentmodelmeasures
2.2.1. Latentinputapproach into the imaging analyses is relatively straightforward. Indeed,
ThegoaloftheLatentInputApproachisaconverseofsortsto multiplemodelmeasurescanbeconsideredsimultaneously.For
theDirectInputApproach.IntheDirectInputApproach,thegoal example,Davis,Love,andPreston(2012)simultaneouslyanalyzed
is to use the neural data N to constrain model mechanisms and cognitive operations related to recognition and representational
θ
parameters thatpredictbehavior.IntheLatentInputApproach, uncertainty by including two related measures in the imaging
thecognitivemodelisusedtoguidetheinferenceofneuraldata analysis from a cognitive model fit to trial-by-trial category
| N,ortomakepredictionsaboutN.Toperformananalysiswithin |     |               |         |                |       |            |             | learningdata. |          |     |                 |     |       |      |         |
| ----------------------------------------------------- | --- | ------------- | ------- | -------------- | ----- | ---------- | ----------- | ------------- | -------- | --- | --------------- | --- | ----- | ---- | ------- |
| this approach,                                        |     | one typically | carries | out            | three | stages,    | illustrated |               |          |     |                 |     |       |      |         |
|                                                       |     |               |         |                |       |            |             | Extensions    | to model |     | discrimination. | One | issue | with | what is |
| in the middle-left                                    |     | panel         | of      | Fig. 1. First, | the   | parameters | of a        |               |          |     |                 |     |       |      |         |
commonlyreferredtoasmodel-basedfMRIisthatmodelstendto
| cognitivemodel |     | θ areestimatedbyfittingthemodeltobehavioral |     |     |     |     |     |              |     |            |           |           |     |           |        |
| -------------- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | ------------ | --- | ---------- | --------- | --------- | --- | --------- | ------ |
|                |     |                                             |     |     |     |     |     | be preferred | to  | the extent | that they | correlate |     | with many | voxels |
dataBalone.Second,theresultingparameterestimatesareused
|     |     |     |     |     |     |     |     | in the brain. | However, |     | it is not clear | that | this | is an appropriate |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | -------- | --- | --------------- | ---- | ---- | ----------------- | --- |
∗
to generate predictions about neural data N , which typically criterion. Because simple cognitive models do not attempt to
representssome‘‘internalstate’’ofthecognitivemodelinterms model every process in the brain, they should not be expected
| of the neural         | measure. |     | Third,                      | one searches | for | correlates | of the |                                                           |     |     |     |     |     |     |     |
| --------------------- | -------- | --- | --------------------------- | ------------ | --- | ---------- | ------ | --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                       |          |     | ∗                           |              |     |            |        | toaccountforthevarianceofeveryvoxel.Furthermore,cognitive |     |     |     |     |     |     |     |
| model’sinternalstateN |          |     | withtheobservedneuraldataN. |              |     |            |        |                                                           |     |     |     |     |     |     |     |
statesmaybecodedbybrainstatesthataredefinedbythepattern
OneexampleofanLatentInputanalysisusingfMRIdatawould
ofactivationovervoxels.Thisnotionofbrainstateismultivariate
beavoxel-by-voxelapplicationofthegenerallinearmodelrelating
∗ as it depends on the pattern of activity, whereas most model-
| themodel’sinternalstateN |     |     | totheneuraldataN(e.g.,O’Doherty, |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
basedanalysesfocusonunivariatecorrelationsbetweenamodel
| Hampton, | & Kim, | 2007). | The | typical result | is a | pattern | of voxels |     |     |     |     |     |     |     |     |
| -------- | ------ | ------ | --- | -------------- | ---- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
measureandanindividualvoxel.
representingsignificantcorrelationswiththecognitivemodel,and
|              |         |       |        |             |            |            |     | One approach   |     | that  | attempts to    | address | these | deficiencies | is     |
| ------------ | ------- | ----- | ------ | ----------- | ---------- | ---------- | --- | -------------- | --- | ----- | -------------- | ------- | ----- | ------------ | ------ |
| these voxels | are     | taken | as the | region of   | the brain  | supporting | the |                |     |       |                |         |       |              |        |
|              |         |       |        |             |            |            |     | model decoding |     | (Mack | et al., 2013). | Rather  | than  | assume       | a sin- |
| mechanism    | assumed |       | by the | model. This | univariate | approach   | is  |                |     |       |                |         |       |              |        |
commonlyreferredtoas‘‘model-basedfMRI’’,butofcourseany gle cognitive model as the ‘‘correct’’ model, this generalization
|     |     |     |     |     |     |     |     | acknowledges | that | there | may be | competing | cognitive |     | models of |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ----- | ------ | --------- | --------- | --- | --------- |
neuralmeasurementcouldbecorrelatedwiththemodelmeasure.
thesamephenomenonandusestheneuraldatatoadjudicatebe-
TheLatentInputApproachiscommonlyusedinreinforcement
tweenthosecompetitors.Itiswellknowninmathematicalpsy-
learningmodelstorelatemechanismsoflearningandprediction
chologythatmodelsassumingverydifferentinternalmechanisms
| errors to | the brain | (e.g., | Gläscher | & O’Doherty, |     | 2010; | Hampton, |     |     |     |     |     |     |     |     |
| --------- | --------- | ------ | -------- | ------------ | --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
cansometimespredictthesameobservedbehavior.Totheextent
| Bossaerts, | & O’Doherty, |     | 2006; | O’Doherty, | Dayan, | Friston, | Critch- |     |     |     |     |     |     |     |     |
| ---------- | ------------ | --- | ----- | ---------- | ------ | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
thatdifferentmodelmechanismsproducedifferentinternalmodel
ley,&Dolan,2003;O’Dohertyetal.,2007),andhasbeenparticu-
|     |     |     |     |     |     |     |     | states, one | way | to discriminate | between |     | models | predicting | the |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------------- | ------- | --- | ------ | ---------- | --- |
larlypowerfulinthefieldofclinicalneuroscience(e.g.,Montague,
samebehavioristocomparethosepredictedinternalmodelstates
Dolan,Friston,&Dayan,2012;Wiecki,Poland,&Frank,2015).One
simple example is the Rescorla–Wagner (RW) model that char- toobservedinternalbrainstates.Modelsthatpredictobservedbe-
acterizestheprocessoflearningaconditionedresponsethrough haviorbutcannotpredictinternalbrainstatesarerejected.
repeatedpresentationsofaconditionedstimulus(Rescorla&Wag- Consider, for example, the work of the Vanderbilt group
|     |     |     |     |     |     |     |     | discussed | earlier | (Palmeri | et al., | 2015; | Purcell | et  | al., 2010, |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------- | -------- | ------- | ----- | ------- | --- | ---------- |
ner,1972).Inthemodel,thevalueoftheunconditionedstimulus
|     |     |     |     |     |     |     |     | 2012). After | excluding |     | neurally-constrained |     | models |     | that could |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --------- | --- | -------------------- | --- | ------ | --- | ---------- |
isrepresentedasu,andthevalueoftheconditionedstimuluson
v not fit the observed behavioral data, they were then able to
| Trialtisrepresentedas |     |     | t .Tolearnthestimulusenvironment,the    |     |     |     |     |             |         |         |        |       |     |     |          |
| --------------------- | --- | --- | --------------------------------------- | --- | --- | --- | --- | ----------- | ------- | ------- | ------ | ----- | --- | --- | -------- |
|                       |     | v   |                                         |     |     |     |     | distinguish | further | between | models | based | on  | how | well the |
| modelassumesthat      |     |     | isupdatedsequentiallyaccordingtoalearn- |     |     |     |     |             |         |         |        |       |     |     |          |
α t ϵ predicted accumulator model dynamics matched the observed
| ingrateparameter |     | ,andanevaluationofthepredictionerror |     |     |     |     |     | .   |     |     |     |     |     |     |     |
| ---------------- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Specifically,afteradecisionismadeandtheunconditionedstimu- neuraldynamicsinmovement-relatedneurons,theneuronsthey
lusispresented,themodel’sinternalstateofthevalueofthecon- hypothesized to carry out an accumulation of evidence (see
ditionedstimulusisupdatedaccordingtotherule alsoPurcell&Palmeri,2017,inthisspecialissue).Onlytheirgated
accumulatormodelproducedaccumulatordynamicsthatmatched
| v =v  | +αϵ. |     |     |     |     |     | (1) |                                                    |     |     |     |     |     |     |     |
| ----- | ---- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| t t−1 |      |     |     |     |     |     |     | theobserveddynamicsofmovement-relatedneuronsinFEF. |     |     |     |     |     |     |     |
Eventually, the internal representation of the value v converges Consider next the recent work of Mack et al. (2013), who
ϵ developed a strategy for evaluating different models of object
| to u, | approaches | zero, | and | the model | ‘‘learns’’ | the | stimulus- |     |     |     |     |     |     |     |     |
| ----- | ---------- | ----- | --- | --------- | ---------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
v categorization on the basis of their consistency with observed
| to-response | pairing. | The | value | of t can | be directly | observed | by  |     |     |     |     |     |     |     |     |
| ----------- | -------- | --- | ----- | -------- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
fMRIdata.Theyspecificallycontrastedtwowell-knowntheories
assessingthestrengthoftheconditionedresponse,whereasother
variables are estimated by fitting the model to behavioral data. ofcategoryrepresentation:exemplarandprototypemodels(see
α
Typically, remains fixed across the trials in an experiment, also Palmeri, 2014). Exemplar models assume that members of
ϵ
allowingonetoderiveatrial-by-trialestimateof throughEq.(1). a category are explicitly stored in memory, and a categorical
Hence,themodelproducestrial-to-trialestimatesofthevalueof decisionforanewstimulusisafunctionofitssimilaritytothese
theconditionedstimulus v andthepredictionerror ϵ .Asoutlined rememberedexemplars.Prototypemodelsassumethatcategory
above, these values can be entered into an fMRI analysis as a representations are abstract, averages of experienced category
time series by convolving them with a hemodynamic response examples, and a categorical decision is a function of similarity
function (HRF), and then regressing the result against the fMRI to the stored category prototypes. In this sense, the prototype

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 71
representation is abstract—a category could be represented in a driven by individual differences in how the brain works. This
locationoffeaturespacethatisnotrepresentativeofanyparticular approachthusstrengthensourunderstandingoftheroleofcertain
known category member. These particular theories of category brainareasincognition,butitalsoaddscredencetothetypeof
representationhavebeenfiercelydebatedfordecades(e.g.,Medin cognitivemodelthatisadoptedtodescribebehavior.
& Schaffer, 1978; Minda & Smith, 2002; Zaki, Nosofsky, Stanton, Intheregressionapproach,parametersofabehavioralmodel
&Cohen,2003).Indeed,intheirfirstanalysis,Macketal.(2013) areusedaspredictorsinaregressionmodeloftheneuralvariables.
showedthatbothexemplarandprototypemodelsprovidednearly In the context of fMRI, behavioral model parameters are often
indistinguishablefitstotheobservedbehavioraldata. enteredasregressorsinagenerallinearmodelthatquantifiesthe
Eventhoughtheexemplarandprototypemodelsmakesimilar BOLDresponseincertainbrainareas(e.g.,Mulder,Wagenmakers,
predictionsaboutbehavior,theydosobyassumingverydifferent Ratcliff, Boekel, & Forstmann, 2012; Summerfield & Koechlin,
kinds of internal representations. Indeed, the degree to which 2010; White et al., 2014). Usually, this is done in addition to
different test items activate these internal representations – regressorsthatrelatetotheexperimentalmanipulations,yielding
similarity to stored exemplars for the exemplar model versus statistical maps of brain activation that reflect the predicted
similaritytocategoryprototypesfortheprototypemodel–differs change in neural activation (i.e., in δ ) for a fixed change in
considerably between the two models. Mack et al. (2013) asked behavioralmodelparameter( θ ),inadditiontothestandardnotion
whether the pattern of brain activity elicited by different test ofachangein δ asafunctionoftheexperimentalmanipulation.
items would be more similar to the pattern of activation of Somepropertiesofbehavioraredifficulttocastinexperimental
internalrepresentationsfortheexemplarmodelortheprototype conditions.Forexample,fluctuationsthatoccuraspartofatime
model.Theyspecificallyevaluatedthemutualinformationshared seriesofobservationsareideallyanalyzedassuch(Wagenmakers,
betweenbrainandmodelstateusingmachinelearningtechniques Farrell, & Ratcliff, 2004). Moreover, these fluctuations may be
like multivariate pattern analysis (MVPA) and representational related to incorrect (Dutilh et al., 2012; Eichele et al., 2008)
similarity analysis (RSA). The patterns of brain activity across or task-unrelated responses, for example due to attentional
trials showed better correspondence to the internal state of the lapses (Mittner et al., 2014; Weissman et al., 2006). For these
exemplarrepresentationthantheprototyperepresentation.These situations it can be useful to study fluctuations in brain and
findings serve as a powerful example of how the neurosciences behaviorovertime.
– combined with a Latent Input Approach – allow us to draw Tounderstandhowthevariabilityinbrainmeasuresfromtrial
conclusionsregardingcompetingcognitivemodelsthatwemight totrialaddstothebehavioralvariability,someresearchershave
nototherwisereach. developedmodelsinwhichparametersareestimatedonatrial-
These model decoding approaches represent an important by-trial basis (Behrens, Woolrich, Walton, & Rushworth, 2007;
departure from the Latent Input Approach discussed above. Brunton,Botvinick,&Brody,2013;Erlich,Brunton,Duan,Hanks,
Namely, these methods do not assume that the model used to & Brody, 2015; Hanks et al., 2015; van Maanen et al., 2011).
interpret the neural data is correct. Instead, they posit a set For example, Behrens et al. (2007) used an optimal model that
of competing models for the underlying cognitive process, and updatestheexpectedrewardforoneoftworesponsesonatrial-
the best explanation is to be determined from each model’s by-trial basis. The parameters of this model were also updated
correspondence to the neural data. Once a cognitive model is on a trial-by-trial basis, based on the actual trial outcome (i.e.,
selected, it can then be used as a lens on the brain data, using thechoiceoftheparticipant)andtheexpectedoutcome(i.e.,the
any existing technique, such as the aforementioned univariate model prediction). Behrens and colleagues found that the level
approachesorrepresentationsimilarityanalysis(RSA).Thisstage atwhichparticipantswereresponsivetochangesintherewards
of the analysis can be seen as confirmatory—the winning model was predictive of anterior cingulate cortex activation on a trial-
hasbeenestablishedandisusedtohelpinterprettheneuraldata. by-trial basis, supporting the idea that anterior cingulate cortex
Pairing model decoding with a model-based analysis approach activation reflects changes in the environment (e.g., Rushworth,
allows for information from brain and behavior to be mutually Mars,&Summerfield,2009).
constrainingthroughthebridgeofthecognitivemodel.Thisextra A slightly different approach was taken by Van Maanen and
stepofselectingamodelbasedonneuraldataisatypicalofLatent colleagues (Boehm et al., 2014; Ho et al., 2012; van Maanen
InputApproaches,andthisstepisnotillustratedinFig.1. et al., 2011). Using the LBA model, these authors estimated
the most likely combination of drift rate and starting point of
2.2.2. Two-stageneuralapproach evidenceaccumulation,giventhedistributionoftheseparameters
The second approach we will discuss that uses behavior across trials. The most likely combination of parameters is
to predict neural data is related to the Two-stage Behavioral determined by the set of parameters that specify the response
Approachdiscussedabove,exceptthathere,theparametersofthe time. While powerful, this method is difficult because the most
behavioralmodel θ areusedtoguidetheanalysisoftheneuraldata likelyparameterestimatesarehighlyuncertain,duetothelarge
Ninsteadofviceversa.Whileasubsetofneuralmodelparameters variabilityinthejointdistributionofthemodelparameters,and
δ couldbereplacedwithasubsetofbehavioralmodelparameters duetothesimplificationofthemodeltoincludeonlytwosources
θ akintotheTwo-stageBehavioralApproach,inpractice,thisis ofvariability.Nevertheless,vanMaanenetal.(2011)showedthat
rarely done. Instead, relationships between θ and δ are formed trial-to-trial fluctuations in BOLD in pre-supplementary motor
through correlational or regression analyses. The correlational area correlated with the trial-to-trial measure of threshold, but
approachhasbeenespeciallysuccessfulinthefieldofperceptual only for speed-stressed trials. This finding was corroborated
decision making (Mulder, van Maanen, & Forstmann, 2014). byBoehmetal.(2014),whofoundasimilarcorrelationbetween
For example, Forstmann et al. (2008), Forstmann et al. (2010), thetrial-to-trialmodelparameterandatrial-to-trialestimateof
and Mansfield, Karayanidis, Jamadar, Heathcote, and Forstmann theContingentNegativeVariation(CNV).TheCNVisaslowrising
(2011) show in various experimental setups that accumulator potential,thoughttorepresentneuralactivationinacortico-basal
model parameters that reflect response caution correlate with ganglia loop including the supplementary/pre-supplementary
averaged BOLD responses in pre-supplementary motor area and motorareas(Nagaietal.,2004;Plichtaetal.,2013).
striatum,tworegionsinthebrainthatarethoughttobeinvolvedin AlthoughtheTwo-stageNeuralApproachhasbeeninstrumen-
mediatingcognitivecontrol.Thesestudiesillustratethatindividual talinelucidatingvariousmechanisticexplanationsofneuraldata,
differencesinbehavior,capturedbyhypothesizedprocesses,are theframeworkneglectsanimportantsourceofconstraint.Namely,

72 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
by analyzing the neural and behavioral data independently, the samepsychologicalprocess.However,thisassumption–knownas
secondary analysis does not statistically guide our understand- stationarity–isastrongone,andisseldomobservedinempirical
| ing of how | these | variables | are related. |     | In this | way, Two-stage |     |     |     |     |     |     |     |     |     |     |
| ---------- | ----- | --------- | ------------ | --- | ------- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
data(e.g.,Craigmile,Peruggia,&Zandt,2010;Peruggia,VanZandt,
frameworks are not statistically reciprocal because the neural &Chen,2002).Turneretal.(2015b)usedamultivariatemodelto
datacannotinfluencetheparameterestimatesofthebehavioral describe the joint activation of a set of brain regions of interest,
model (cf. Forstmann et al., 2011). To accomplish such a goal, a andusedthisdescriptiontoenhancetheclassicDDM.Inacross
frameworkwouldneedtoautomaticallylearnthecovariationof validation test, they showed that their extended model could
theneuralandbehavioralparametersinharmonywiththeanaly- generatebetterpredictionsaboutbehavioraldatathantheDDM
sisoftheneuralandbehavioraldata.Suchaframeworkisthetopic alone,demonstratingthatneurophysiologycanbeusedtoimprove
discussedinthenextsection:SimultaneousModeling. explanationsabouttrial-to-trialfluctuationsinbehavior.
|     |     |     |     |     |     |     |     | Effectively, |     | the Joint | Modeling | Approach |     | is a | strategy | for |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --------- | -------- | -------- | --- | ---- | -------- | --- |
2.3. Simultaneousmodeling treatinggroupsofparametersascovariates,andthiscovariation
|     |     |     |     |     |     |     |     | is learned | through | hierarchical |     | modeling. | However, |     | one | could |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------- | ------------ | --- | --------- | -------- | --- | --- | ----- |
At this point, we have discussed two general analytic ap- imagine an approach for performing model-based cognitive
proachesthatapplyunidirectionalstatisticalinfluence:modeling neuroscience that is similar to the Two-stage Neural approach,
andanalysisofonesourceofdataguidesthemodelingandanaly- butinsteadofcorrelatingorregressingvariablesafterindependent
sisofanothersource.Theprimarymotivationoftheseapproaches analyses,theparametersoftheregressionequationareestimated.
is that one measure is particularly well suited for answering a SuchanapproachcanbethoughtofasaJointModelingApproach,
key theoretical question. In this way, one measure carries more Ω
|               |              |     |          |        |          |      |        | except the    | linking | parameters |      | are          | deterministic. |     | Recently, |      |
| ------------- | ------------ | --- | -------- | ------ | -------- | ---- | ------ | ------------- | ------- | ---------- | ---- | ------------ | -------------- | --- | --------- | ---- |
| ‘‘theoretical | importance’’ |     | than the | other. | However, | some | model- |               |         |            |      |              |                |     |           |      |
|               |              |     |          |        |          |      |        | this approach |         | has been   | used | in cognitive | neuroscience   |     | to        | link |
ingapproachesareagnosticinspecifyingwhichmeasureismore
|     |     |     |     |     |     |     |     | decision | models | to  | neural fluctuations. |     | For | example, | Nunez, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------ | --- | -------------------- | --- | --- | -------- | ------ | --- |
important,andinsteadpositabidirectionallinkbetweenthetwo
|     |     |     |     |     |     |     |     | Srinivasan, | and | Vandekerckhove |     | (2015) | used | EEG | data | on a |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------------- | --- | ------ | ---- | --- | ---- | ---- |
measures.Similartothesubdivisionsinotherresearchgoalsabove,
perceptualdecisionmakingexperimentasaproxyforattention.
thelevelatwhichthelinkisestablishedisanimportantdistinction
Theycontrolledtherateofflickeringstimulipresentedtosubjects
betweenthetwoapproaches,whichwewillnowdiscussinturn. to match the sampling rate of their EEG data, a measure known
|     |     |     |     |     |     |     |     | as the steady-state |     | visual | evoked | potential. |     | Importantly, | Nunez |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ------ | ------ | ---------- | --- | ------------ | ----- | --- |
2.3.1. Jointmodelingapproach
|     |     |     |     |     |     |     |     | et al. (2015) |     | showed | that individual |     | differences |     | in attention |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------ | --------------- | --- | ----------- | --- | ------------ | --- |
ThenextapproachwediscussistherecentlydevelopedJoint
|     |     |     |     |     |     |     |     | or noise | suppression |     | was indicative |     | of the | choice | behavior, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | -------------- | --- | ------ | ------ | --------- | --- |
Modelingframework(Turner,2015;Turneretal.,2013b,2015b).
|     |     |     |     |     |     |     |     | specifically | it resulted |     | in faster | responses | with | higher | accuracy. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----------- | --- | --------- | --------- | ---- | ------ | --------- | --- |
TheJointModelingApproachisconceptuallysimilartotheTwo- In a particularly novel application, Frank et al. (2015) showed
stageNeuralApproachinthatitattemptstorelatetheparameters how models of reinforcement learning could be fused with the
of the behavioral model to the parameters of the neural model. DDM to gain insight into activity in the subthalamic nucleus
However, statistically speaking, the Joint Modeling Approach is (STN). In their study, Frank et al. (2015) used simultaneous EEG
unique in the way it bridges this connection. Specifically, it and fMRI measures as a covariate in the estimation of single-
| assumes            | an overarching |                 | distribution | that       | enforces    | an  | explicit  |                   |           |               |                  |           |             |       |              |     |
| ------------------ | -------------- | --------------- | ------------ | ---------- | ----------- | --- | --------- | ----------------- | --------- | ------------- | ---------------- | --------- | ----------- | ----- | ------------ | --- |
|                    |                |                 |              |            |             |     |           | trial parameters. |           | Specifically, |                  | they used | pre-defined |       | regions      | of  |
| connection         | between        | these           | parameters.  | The        | bottom-left |     | panel of  |                   |           |               |                  |           |             |       |              |     |
|                    |                |                 |              |            |             |     |           | interest          | including | the           | presupplementary |           | motor       | area, | STN,         | and |
| Fig. 1 illustrates |                | this connection | via          | the        | parameters  | Ω   | that link |                   |           |               |                  |           |             |       |              |     |
|                    |                |                 |              |            |             |     |           | a general         | measure   | of            | mid-frontal      | EEG       | theta       | power | to constrain |     |
| θ to δ .           | In this        | illustration,   | the          | connection | enforced    |     | by Ω is   |                   |           |               |                  |           |             |       |              |     |
trial-to-trialfluctuationsinresponsethreshold,andBOLDactivity
clearlyabstract;onemustmakeaspecificassumptionabouthow
|           |          |          |                  |             |        |                |          | in the caudate |            | to constrain | trial-to-trial |             | fluctuations |               | in evidence    |      |
| --------- | -------- | -------- | ---------------- | ----------- | ------ | -------------- | -------- | -------------- | ---------- | ------------ | -------------- | ----------- | ------------ | ------------- | -------------- | ---- |
| θ and δ   | should   | coexist  | in their         | explanation | of     | the underlying |          |                |            |              |                |             |              |               |                |      |
|           |          |          |                  |             |        |                |          | accumulation.  |            | Their        | work is        | important   | because      |               | it establishes |      |
| cognitive | process. | As an    | example,         | one         | simple | linking        | function |                |            |              |                |             |              |               |                |      |
|           |          |          |                  |             |        |                |          | concrete       | links      | between      | STN            | and pre-SMA |              | communication |                | as a |
| used in   | practice | has been | the multivariate |             | normal | distribution   |          |                |            |              |                |             |              |               |                |      |
|           |          |          |                  |             |        |                |          | function       | of varying | reward       | structure,     |             | as well      | as            | a model        | that |
Ω
| where | consists | of the | hyper | mean | vector | and the | hyper |     |     |     |     |     |     |     |     |     |
| ----- | -------- | ------ | ----- | ---- | ------ | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
usesfluctuationsindecisionconflict(asmeasuredbymultimodal
variance–covariancematrix.Thisconnectionisimportantbecause
|                                                        |     |     |     |     |     |     |          | activity                     | in the | dorsomedial |     | frontal cortex) |     | to adjust | response |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | -------- | ---------------------------- | ------ | ----------- | --- | --------------- | --- | --------- | -------- | --- |
| itallowstheinformationcontainedintheneuraldataN        |     |     |     |     |     |     | toaffect |                              |        |             |     |                 |     |           |          |     |
|                                                        |     |     |     |     |     |     | θ        | thresholdfromtrial-to-trial. |        |             |     |                 |     |           |          |     |
| theinformationwelearnaboutthebehavioralmodelparameters |     |     |     |     |     |     |          | .                            |        |             |     |                 |     |           |          |     |
ThemajorlimitationoftheJointModelingApproachisitscom-
PerhapsthegreatestbenefitoftheJointModelingApproachis
plexity,whichhindersourabilitytousetheapproacheffectively
itsflexibility—itcanbeappliedtodifferentmodalities(e.g.,fMRI
|               |         |                 |             |                |                   |                |        | in two ways. | First,          | to      | estimate    | all of    | the model | parameters, |            | we    |
| ------------- | ------- | --------------- | ----------- | -------------- | ----------------- | -------------- | ------ | ------------ | --------------- | ------- | ----------- | --------- | --------- | ----------- | ---------- | ----- |
| or EEG data), | make    | different       | assumptions |                | about             | the underlying |        |              |                 |         |             |           |           |             |            |       |
|               |         |                 |             |                |                   |                |        | must use     | a sophisticated |         | system      | of Markov |           | chain       | Monte      | Carlo |
| cognitive     | process | (i.e., changing |             | the behavioral |                   | submodel),     | and    |              |                 |         |             |           |           |             |            |       |
|               |         |                 |             |                |                   |                |        | sampling     | with            | updates | on separate | blocks    | of        | model       | parameters |       |
| establish     | a link  | at any number   | of          | levels         | in a hierarchical |                | model. |              |                 |         |             |           |           |             |            |       |
For example, Turner et al. (2013b) used structural diffusion (seeTurner,2015;Turneretal.,2013b;Turner,Sederberg,Brown,
weighted imagining data to explain differences in patterns of &Steyvers,2013c;Turneretal.,2015b,fordetails).Thisinvolves
choice response time data across subjects. They showed how a derivingtheconditionaldistributionofblocksofparameters,and
jointmodelequippedwithinformationabouttheinterconnectivity ifdesired,establishingconjugaterelationshipsbetweentheprior
andposteriorforeffectiveestimation.Oneexampleofthishasbeen
| of important | brain      | areas       | could make | accurate | predictions |               | about |         |                   |     |        |            |     |         |        |     |
| ------------ | ---------- | ----------- | ---------- | -------- | ----------- | ------------- | ----- | ------- | ----------------- | --- | ------ | ---------- | --- | ------- | ------ | --- |
|              |            |             |            |          |             |               |       | the use | of a multivariate |     | normal | assumption |     | to link | neural | and |
| a subject’s  | behavioral | performance |            | in the   | absence     | of behavioral |       |         |                   |     |        |            |     |         |        |     |
behavioralsubmodelparameters(Turneretal.,2013b,2015b).In
| data. Turner | et                 | al. (2015b) | extended |      | this approach |      | to build |                |     |             |     |            |         |     |               |     |
| ------------ | ------------------ | ----------- | -------- | ---- | ------------- | ---- | -------- | -------------- | --- | ----------- | --- | ---------- | ------- | --- | ------------- | --- |
|              |                    |             |          |      |               |      |          | this approach, |     | an increase | in  | any neural | measure |     | automatically |     |
| in brain     | state fluctuations |             | measured | with | fMRI          | into | the DDM. |                |     |             |     |            |         |     |               |     |
The problem Turner et al. (2015b) addressed centered on a scalestheincreaseinthebehavioralmodelparameters,andvice
lack of information about within-trial accumulation dynamics. versa.Second,agreatdealofdatamustbeavailabletoappreciate
In behavioral choice response time experiments, following the themagnitudeoftheeffectsofinterest.Thisresultisdrivenbya
presentation of a stimulus, researchers can only observe the complexity/flexibilitytradeoffwediscussbelow,butthebasicidea
eventual choice and response time. These data are then used to is that as the number of parameters increases, the influence the
estimateparametersofacognitivemodel,followinganassumption datacanhaveonthejointposteriordistributiondecreases.When
that the data observed on each of these trials arises from the amodeliscomplexrelativetothedata,onesimpleapproachto

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 73
reducethecomplexityistoreducethenumberofmodelparame- analyses (Anderson, 2007), but recent work has taken these ex-
ters(Myung&Pitt,1997).InhierarchicalmodelsliketheJointMod- plorationstogenerateout-of-sample,confirmatorypredictionsfor
elingApproach,onewaytoaccomplishthisistoreducethenumber neuraldata.Intheseconfirmatorystudies,thespecificpatternof
θ
oflevelsinthehierarchybyremovingitssubmodels(i.e.,models moduleactivations(i.e.,theparameters )inthemodelsimulta-
withintheJointModelthatexplainonesubsetofthedata).Sucha neouslyaffectsthemodel’spredictionsfortheBOLDresponseand
strategyconstitutesourfinalapproach:theIntegrativeapproach. thebehavioraloutcome.Althoughglobal,whole-brainpredictions
|     |     |     |     |     |     |     | could be | made within | this | framework, | the | strict assumption |     | of  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ---- | ---------- | --- | ----------------- | --- | --- |
localizedmoduleactivityinthebrainconstitutesafullyconfirma-
2.3.2. Integrativeapproach
toryIntegrativeapproach,wherepredictionsforneuralactivity–
| In the | Integrative | approach, |     | the goal | is to develop | a single |     |     |     |     |     |     |     |     |
| ------ | ----------- | --------- | --- | -------- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
aswellasbehavioraldata–canbequantitativelyevaluated.
cognitivemodelcapableofpredictingbothneuralandbehavioral
TheACT-Rframeworkprovidesanuniqueperspectiveonper-
measures.Thisapproach,illustratedinthebottom-rightpanelof
formingtheintegrationbetweenneuralandbehavioralmeasures,
| Fig. 1, uses | one set | of parameters |     | θ to explain | the | neural N and |     |     |     |     |     |     |     |     |
| ------------ | ------- | ------------- | --- | ------------ | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
butactuallytestingthesemodelsisnontrivial.Themajorlimita-
behavioral B data jointly. Notice that the Integrative approach tionisthatonemustassumeasetofspecificmodules,andtheac-
differsfromtheJointModelingApproachbecausetheparameters tivationofthesemodulesinthebehavioralmodelislatent,which
θ
are directly connected to the data—there is no overarching makestheiractivationdifficulttoidentifyinbehavioraldata.Al-
distributionΩ
tointervenebetweenthedatasources.Integrative
|     |     |     |     |     |     |     | though neural | data | facilitate | this | identification | process, | current |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | ---------- | ---- | -------------- | -------- | ------- | --- |
approachesallowtheneuraldataNtohaveagreaterinfluenceon
|     |     |     |     |     |     |     | solutions | rely heavily | on  | assumptions | about | how | modules | are |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------------ | --- | ----------- | ----- | --- | ------- | --- |
thebehavioraldataB,astatisticalpropertythatcanbemeasured
representedinpatternsofneuralactivity(Anderson,2012).Fur-
bymutualinformation.
|     |     |     |     |     |     |     | thermore, | it is unclear | how | one | would objectively |     | decompose |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------------- | --- | --- | ----------------- | --- | --------- | --- |
Oftheapproacheswehavediscussed,theIntegrativeapproach
othercognitivemodelsintoadiscretesetofmoduleswhilepre-
| is arguably | the most | difficult | to  | develop. | Its use requires | strong |     |     |     |     |     |     |     |     |
| ----------- | -------- | --------- | --- | -------- | ---------------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
servingtheirkeytheoreticalandconvenientproperties(forexam-
commitments to both the underlying cognitive process and plesofcognitivemodelsinthestyleofACT-R,seevanMaanen&
where this process is executed in the brain. One technical Van Rijn, 2010; van Maanen, Van Rijn, & Borst, 2009; van Maa-
hurdleinusinganIntegrativeapproachliesinthedescriptionof nen,VanRijn,&Taatgen,2012).Forexample,theLinearBallistic
randomvariableswithdifferenttemporalproperties.Forexample, Accumulator(LBA;Brown&Heathcote,2008)modelhasenjoyed
neurophysiologicalmeasuresaretypicallyobservedonamoment- widespread success due to its parsimony and remarkable math-
by-moment basis, detailing activation in the brain throughout ematicaltractability.BreakingtheLBAmodeldownintoitscon-
thetrial.Bycontrast,behavioraldataaretypicallyobservedonly
stituentpartscouldcompromisethistractabilityinsuchawaythat
| at the end | of a | trial, | such as | in any | number | of perceptual |     |     |     |     |     |     |     |     |
| ---------- | ---- | ------ | ------- | ------ | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
estimationofthemodel’sparameterswouldbenontrivial.Hence,
| decision | making | tasks. | So, in | the instantiation |     | of a cognitive |     |     |     |     |     |     |     |     |
| -------- | ------ | ------ | ------ | ----------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
itisclearthatnoteverycognitivemodelcaneasilybetransformed
theory that uses the Integrative approach, we would need a andpreparedforananalysisusingtheIntegrativeApproach.Atthis
moment-by-moment prediction of neural data, and a trial-by- point,anaturalquestiontoaskis,underwhatconditionsshouldan
trialpredictionofthebehavioraldata,usuallyassumedtobethe approachbeusedforananalysis?
| result of  | a series  | of unobservable |             | (i.e.,      | latent) processes. | Given   |     |     |     |     |     |     |     |     |
| ---------- | --------- | --------------- | ----------- | ----------- | ------------------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| the unique | structure | of              | Integrative | approaches, | properly           | fitting |     |     |     |     |     |     |     |     |
3. Comparingtheapproaches
| them to | data is | a difficult | task, | often | involving | sophisticated |     |     |     |     |     |     |     |     |
| ------- | ------- | ----------- | ----- | ----- | --------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
techniquessuchasHiddenMarkovModels(e.g.,Anderson,2012;
Itisimportanttosupplementourdiscussionofapproachesto
Anderson,Betts,Ferris,&Fincham,2010),orBayesianchangepoint
|     |     |     |     |     |     |     | model-based | cognitive | neuroscience |     | with a | guide | to how | these |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | ------------ | --- | ------ | ----- | ------ | ----- |
analyses(e.g.,Mohammad-Djafari&Féron,2006).
|     |     |     |     |     |     |     | approaches | compare. | This | comparison | is difficult |     | and likely | to  |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ---- | ---------- | ------------ | --- | ---------- | --- |
Some recent applications of ACT-R have aimed for this Inte- be highly subjective. How should the various approaches be
grativeApproach.ACT-Rassumesthepresenceofdistinctcogni- evaluated? Along what dimensions should they be compared
tive modules that are recruited sequentially during a task. The and contrasted? Do these approaches cover all possible types of
recruitmentofthesemodulesacrossthetimecourseofthetask linkagebetweenneuralandbehavioralmeasures?Despiteourfear
canberepresentedasavectorofbinaryoutcomes,suchthata1 of improperly considering these questions, we will persist and
indicatesthatamoduleisbeingused,anda0indicatesitisnot attempttoorganizethesixcoreapproachesdiscussedinthisarticle
beingused.Thisvectornaturallylendsitselftoconvolutionwith along dimensions that are relevant for practical implementation
thecanonicalHRFinthesamewayasexperimentaldesignvari- (notethatwehavegroupedbothtypesofTwo-Stageapproaches
| ables. The | result | of the | convolution | is a | model-generated | BOLD |     |     |     |     |     |     |     |     |
| ---------- | ------ | ------ | ----------- | ---- | --------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
togetherforthisdiscussion).Table1providesalistofkeyfactors
signal that can be compared to empirical data. In this way, the thatcanbeusedtocomparethestrengthsandweaknessesofthe
| ACT-Rmodelcanactuallybeusedinbothexploratoryandconfir- |     |     |     |     |     |     | approaches. |     |     |     |     |     |     |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
matoryresearch.Whenusedforexploration,themodel-generated
BOLDsignalisregressedagainstthedatainavoxel-by-voxelfash- 3.1. Numberofstages
| ion through | the | general | linear | model (Borst | & Anderson, | 2013; |     |     |     |     |     |     |     |     |
| ----------- | --- | ------- | ------ | ------------ | ----------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
Borst, Taatgen, & Van Rijn, 2010b). From this analysis, clusters The first factor we could compare the approaches on is the
| of voxels | typically | emerge, | and | these clusters | are | taken to rep- |     |     |     |     |     |     |     |     |
| --------- | --------- | ------- | --- | -------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
numberofprocessingstages.Thefewestnumberofstagesoccur
| resent brain     | areas | where    | the modules |                   | are physically | executed.      |          |          |                |     |               |                 |             |     |
| ---------------- | ----- | -------- | ----------- | ----------------- | -------------- | -------------- | -------- | -------- | -------------- | --- | ------------- | --------------- | ----------- | --- |
|                  |       |          |             |                   |                |                | when the | approach | considers      |     | both measures | simultaneously. |             |     |
| This explorative |       | analysis | more        | closely resembles |                | the Latent In- |          |          |                |     |               |                 |             |     |
|                  |       |          |             |                   |                |                | Because  | both the | Joint Modeling |     | Approach      | and the         | Integrative |     |
put Approach. However, the ACT-R model can also be used in a approach make formal assumptions about how both behavioral
confirmatoryfashion(Anderson,2007;Anderson,Byrne,Fincham, andneuralmeasuresarise,afullcomputationalmodelisfittothe
&Gunn,2008a;Anderson,Fincham,Qin,&Stocco,2008b;Borst, entiresetofdatainonestage.Anotherapproachrequiringonlyone
Taatgen, Stocco, & Van Rijn, 2010a). To do this, Anderson and stageistheDirectInputApproach,wheretheneuraldatareplace
colleagues have identified which brain areas should become ac- dynamicsofthebehavioralmodel.Here,onlythebehavioraldata
tiveduringtherecruitmentofdifferentmodules(Andersonetal., areconsideredwhilefittingthemodeltodata,butthisprocessstill
2008b;Borst,Nijboer,Taatgen,VanRijn,&Anderson,2015).These onlyrequiresasingleprocessingstage.TheLatentInputandTwo-
brain areas were identified primarily from several exploratory stageapproachestypicallyrequirethegreatestnumberofstagesat

74 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
Table1
Acomparisonbetweenthesixdifferentanalyticapproachesonfiveimportantfactors.Notethatthesedescriptionshavebeenformedontheworkwearefamiliarwith,and
thefactorsrepresentconsiderationsthatareespeciallyimportanttous.
Factor Theoretical Two-stage DirectInput LatentInput JointModeling Integrative
Numberofstages {2,3,...} {2,3} 1 {2,3} 1 1
Commitmenttoaparticulartheory None Weak Medium Weak Weak Strong
Typeofinformationflow Conceptual One-way One-way One-way Two-way Two-way
Difficultyofimplementation High Low Medium Medium High High
Typeofexploration Exploratory Exploratory Confirmatory Either Either Confirmatory
twoorsometimesthree.Ifaseparatesimulationstageisrequired we refer to this type of information flow as two-way because
∗
togenerateneuralpredictionsN ,LatentInputApproacheshave information flows in both directions. When one source of data
three stages, whereas if the internal state of a model can be enforces direct constraint on the other measure, we refer to
directly inferred when the behavioral model is fit to behavioral this type of information flow as one-way. All of the remaining
data(e.g.,asinthereinforcementmodelsdescribedabove),then approaches use information flow that is one-way to maximize
theLatentInputApproachonlyrequirestwostages.IntheTwo- constraintintheirmodels.
stage approach, if the parameters of the behavioral model can While on the surface, a one-way information flow may seem
beregressed(orcorrelatedwith)therawneuraldata,thenonly like a weakness, there are sometimes important theoretical
two stages are required. However, if some preliminary analyses reasons for enforcing this strict directionality. Consider, for
of the neural data are required, then the Two-stage approach example, the illustrated uses of the Latent Input Approach for
will require three stages. Finally, the Theoretical Approach can model discrimination (Mack et al., 2013; Palmeri, 2014; Palmeri
require anywhere from two to an infinite number of stages. In etal.,2015;Purcelletal.,2010,2012).Herethegoalwastouse
the simplest scenario, the first stage consists of observing some neural data to help discriminate between models of perceptual
patternorphenomenaofinterestintheneuraldata,andthesecond decisionmakingormodelsofcategorizationthatmakethesame
stageconsistsofthedevelopmentofabehavioralmodel.However, behavioralpredictions.Themodelswerefittothebehavioraldata
TheoreticalApproachescanalsobecomplextoimplementbecause inexactlythesamewaytheymightbefitifneuraldatawerenot
they can involve an extensive, iterative process of running new evenconsidered.Nocompromisesweremadeinthebehavioralfits
experiments and refining a developing model (Shiffrin & Nobel, totakeintoaccounttheneuraldata,asmightbethecaseforthe
1997). Joint Modeling or Integrative Approaches. Only after the models
werefittothebehavioraldatawerethepredictedinternalstates
3.2. Commitmenttoaparticulartheory ofthemodelthencomparedtoobservedneuralstatesinthebrain.
Finally,modelswererejectediftheycouldnotadequatelycapture
The second factor involves the role of flexibility in applying thoseobservedneuralstatesinthebrain.
newtheoriestothedata.Forexample,weconsidertheTwo-stage
Approach to have weak commitment to any particular theory: 3.4. Difficultyofimplementation
one could freely use the same procedure to test any number of
behavioral models on the same neural data. The commitment Fromapragmaticperspective,itisalsoimportanttoconsider
to a particular theory is similarly weak in the Latent Input and the difficulty of performing analyses with these six approaches.
Joint Modeling Approaches, where behavioral models can easily Perhapstheeasiestapproachtoimplementforthereadersofthis
beswitchedoutandfitstodatacompared.WeconsidertheDirect special issue is the Two-stage Approach, where the parameters
InputApproachtobemorecommittedtoaparticulartheorythan ofacognitivemodelaresimplyregressedagainstaneuralsignal
theseaforementionedapproaches.Forexample,whilePurcelletal. of interest. Of medium difficulty are the Direct Input and Latent
(2010) used neural data to test different assumptions about the InputApproaches,becausetheyoftenrequiremodelsimulations
accumulationprocess,theystillmaintainedacommitmenttothe or additional theoretical overhead to fit the models to data.
sequential sampling framework for these models. In this way, The Joint Modeling and Integrative Approaches are considered
their analysis relies on some theoretical assumptions about the difficult to implement because they either require sophisticated
accumulation process, but not in a way that is inflexible. Going partitioningoftheparameterspace(e.g.,Turneretal.,2015b),or
onestepbeyondthisistheIntegrativeApproach,whichrequires estimation of hidden Markov model parameters (e.g., Anderson,
strongcommitmentstoaparticularmodelingframeworksuchas 2012;Andersonetal.,2010).Perhapsthemostdifficultapproach
in Anderson and colleagues’ work (e.g., Anderson et al., 2008b; to implement is the Theoretical Approach, where models must
Borst et al., 2015). In this approach, it is difficult to imagine be carefully constructed and iteratively fit to data as a test of
testing different models that are not contained within a similar specificassumptions.Tomakemattersworse,thereisnoclearend
overarching theory. Finally, the Theoretical Approach makes no pointwhendevelopinganewcognitivemodelintheTheoretical
commitmenttoanyparticulartheory,insteaditusesthedatato Approach.
guidethedevelopmentofthetheoryitself.
3.5. Typeofexploration
3.3. Typeofinformationflow
Afinalconsiderationisthetypeofexplorationthatcanbeused
Another factor to consider is the type of information flow. In underaspecificapproach.Approachescanbeusedforexploratory
Table1,weconsiderthreetypes:conceptual,one-way,andtwo- or confirmatory purposes, or some mixture of the two. The
way.IntheTheoreticalApproach,theneuraldatacanonlyguide TheoreticalandTwo-stageApproachesareconsideredexploratory
thedevelopmentofthebehavioralmodelconceptually—thereisno becausethegeneralstrategyinvolvesasequenceoftests,iterating
formalrelationshipbetweenthebehavioralandneuralmeasures. toward a solution or explanation of the data. The Direct Input
At the other extreme, both the Joint Modeling and Integrative Approach is considered a confirmatory approach because the
approaches use the information contained in either measure to neural data are used to directly replace certain mechanisms in
directlyconstraintheestimatesofthemodels’parameters.Hence, the model, providing a test of the neural measure’s plausibility

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 75
in predicting the behavioral response. The Integrative Approach theencodingcomponentinourmodeltosearchfortemporally-
isalsoconfirmatorybecauseitmakesspecificassumptionsabout relatedactivationsinthebrain.Bothoftheseanalysesconstitute
how both measures arise, where good fits to data support the searches through our neural data as a way to better understand
assumptionsofthemodel,andpoorfitsrefutethem.Weregardthe howthebrainproducesbehaviorfromamechanisticperspective.
LatentInputApproachasbeingexploratorywhenusedinatypical Inthisway,theseanalysesareunidirectionalanddonotvalidate
‘‘model-based’’analysis,butconfirmatorywhenusedtocompare orconfirmourmodel,butthisisperfectlyacceptablebecauseitis
modelstooneanotherasinMacketal.(2013)andPurcelletal. consistentwithourcurrentgoals.
(2012).Inthisway,theLatentInputApproachislistedas‘‘either’’ Ourexploratoryanalyseshavepavedthewayforsubsequent
because the specific usage depends on the situation. Finally, the investigations,andnowsupposewewishtousetheneuraldata
JointModelingApproachisalsoconsideredbothconfirmatoryand to better constrain our behavioral model. We now have well-
exploratory, because its usage depends on the how the linking defined hypotheses about which brain areas are involved in
functionisspecified.Forexample,onecoulduseagenerallinear stimulusencoding,andwesuspectthatthesystematicactivations
modelasthelinkingfunction–aconfirmatoryapproach–orone inthesebrainareashaveacorrespondencetotheencodingphase
could use ambiguous priors on hyperparameters that specify a of our model. At this point, we must reconsider our specific
multivariateGaussianlinkingfunction–anexploratoryapproach. goals.Ifthegoalofouranalysisistopredictbehavior,wemight
| Furthermore, | the | specific | prior | used | on the | hyperparameters |     |         |        |                |     |                    |     |        |     |
| ------------ | --- | -------- | ----- | ---- | ------ | --------------- | --- | ------- | ------ | -------------- | --- | ------------------ | --- | ------ | --- |
|              |     |          |       |      |        |                 |     | use the | Direct | Input Approach |     | to map activations |     | in the | key |
allowstheJointModelingApproachtomixbetweenconfirmatory brain areas directly to the encoding component of our model.
andexploratoryrolesinananalysis. By contrast, if our goal is to infer relationships between the
neuralandbehavioralmeasures,wemightusetheJointModeling
4. Choicesandlimitations Approach to test specific impositions of brain activations to the
|     |     |     |     |     |     |     |     | parameters | regulating | the | encoding | process | in our | model. | Both |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---------- | --- | -------- | ------- | ------ | ------ | ---- |
In this article, our goal was to highlight and discuss the of these approaches are more confirmatory because they rely
prominent approaches to analysis in the emerging subfield of on specific hypotheses and assumptions that were derived from
|             |             |               |          |          |       |         |         | our exploratory |        | analyses;    | however,  | they    | still | only guide | our    |
| ----------- | ----------- | ------------- | -------- | -------- | ----- | ------- | ------- | --------------- | ------ | ------------ | --------- | ------- | ----- | ---------- | ------ |
| model-based | cognitive   | neuroscience. |          | However, |       | we have | not yet |                 |        |              |           |         |       |            |        |
|             |             |               |          |          |       |         |         | inference.      | In the | Direct Input | analysis, | because |       | our goal   | was to |
| provided    | a guideline | for           | choosing | between  | them, | nor     | have we |                 |        |              |           |         |       |            |        |
predictthebehavioraldata,wehavecompromisedourabilityto
discussedingreaterdetailthelimitationsofchoosingaparticular
approach.Inthissection,wewilladdressbothoftheseissues. evaluate the model’s suitability for the neural data. We cannot
|     |     |     |     |     |     |     |     | make predictions |     | about | neural | data that | we have | conditioned |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----- | ------ | --------- | ------- | ----------- | --- |
4.1. Choosingbetweenapproaches on, as so we cannot evaluate how well the model captures
|     |     |     |     |     |     |     |     | these aspects | of  | our (neural) | data. | On the | other | hand, the | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------ | ----- | ------ | ----- | --------- | ----- |
ModelingApproachattemptstocapturebothaspectsofthedata
Althoughwehavedescribed,compared,andcontrastedsixim-
simultaneously,andasaresult,itspredictionsforthebehavioral
portantapproachesforanalysis,wehavenotprovidedaguideline
|     |     |     |     |     |     |     |     | data are | compromised | by  | the model’s | obligations |     | to the | neural |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | ----------- | ----------- | --- | ------ | ------ |
forhowtheseapproachescouldbeusedtoadvancepsychological
|     |     |     |     |     |     |     |     | data. Because | the | Joint | Modeling | Approach | does | not explicitly |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | -------- | -------- | ---- | -------------- | --- |
theory.Webelievethateachoftheseapproacheshavetheirown
|     |     |     |     |     |     |     |     | condition | on either | variable, | it  | can reveal | interesting | generative |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --------- | --- | ---------- | ----------- | ---------- | --- |
utilityinthepursuitanddevelopmentofcomputationalmodels,
|     |     |     |     |     |     |     |     | properties | of our | model, | but its | discriminative |     | (i.e., predictive) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ------ | ------- | -------------- | --- | ------------------ | --- |
andtheprimaryfactorinchoosingbetweenthemisthegoalofthe
powerissomewhatdiminished(Bishop&Lasserre,2007).
analysis.Furthermore,asatheoryprogresses,itisimportanttore-
Atthispoint,wehavenowdevelopedourmodelandevaluated
alizethatthegoalsofananalysisshouldchange.Tothisend,wead-
vocateusingalloftheseapproachestomovefromanexploratory the relationships between brain and behavior in a variety of
analysistoaconfirmatoryone. analytic approaches. We know better than anyone in the world
Toseehowthiswouldworkinpractice,considerthefollowing wheretheencodingpartofourmodeliscarriedoutinthebrain,
stagesofmodeldevelopment.Intheinitialstages,oneapproach andhowdifferencesinthepatternofactivationinthesebrainareas
is to develop a cognitive theory by acknowledging patterns in contributetobehavioraldifferences.Asafinaltestandvalidation
ofourmodel,wecannowmovetothemostconfirmatoryanalysis
| the data | from both | the | brain | and the | behavior. | For | example, |     |     |     |     |     |     |     |     |
| -------- | --------- | --- | ----- | ------- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
wehavediscussedhere:theIntegrativeApproach.Toestablishan
knowingthatthebrainmustfirstencodestimulusinformationin
integrativemodel,wemustfirstmakesomespecificassumptions
| lower-level | visual    | areas | before | a representation |     | of the | stimulus  |           |             |        |           |          |           |              |     |
| ----------- | --------- | ----- | ------ | ---------------- | --- | ------ | --------- | --------- | ----------- | ------ | --------- | -------- | --------- | ------------ | --- |
|             |           |       |        |                  |     |        |           | about how | activations | in     | key brain | areas    | map to    | the encoding |     |
| can be      | perceived | and   | acted  | upon could       | be  | used   | to impose |           |             |        |           |          |           |              |     |
|             |           |       |        |                  |     |        |           | component | of our      | model. | This      | can be a | difficult | process,     | but |
orderinabehavioralmodel.Suchknowledgemightmotivatethe
supposefornowthatwehaveformallyarticulatedthismapping
| development | of a | visual | encoding | component |     | of the | model that |     |     |     |     |     |     |     |     |
| ----------- | ---- | ------ | -------- | --------- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
precedesthedevelopmentofanaccuratestimulusrepresentation. in our model, derived from our previous exploratory analyses.
Instantiation of the encoding process in the behavioral model Our goal now is to show that this integrative version of our
is an implementation of the Theoretical Approach, because the model can produce patterns of data that match all aspects of
development is motivated by brain data. Here, our goal was to our data. That is, adjustments of one model parameter should
simplydevelopamodelthatabidesbycertainphysiologicaltiming make specific predictions about how the pattern of neural and
restrictions as a way to establish a more constrained stimulus behavioral measures changes, and ideally, how these changes
|     |     |     |     |     |     |     |     | could be | selectively | influenced |     | experimentally | (e.g., | Heathcote, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ---------- | --- | -------------- | ------ | ---------- | --- |
processingorder.
|     |     |     |     |     |     |     |     | Brown, & | Wagemakers, |     | 2015). | In our opinion, |     | this integrative |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | ------ | --------------- | --- | ---------------- | --- |
Afterthedevelopmentofthemodel,ourgoalshaveadvanced—
analysisrepresentsthestrongesttestofpsychologicaltheory,but
supposewenowwishtoidentifywherethisencodingcomponent
|              |            |     |          |               |       |       |        | such a test | would | be misguided |     | if not first | informed | by the | less |
| ------------ | ---------- | --- | -------- | ------------- | ----- | ----- | ------ | ----------- | ----- | ------------ | --- | ------------ | -------- | ------ | ---- |
| of our model | is carried |     | out, and | specifically, | which | areas | of the |             |       |              |     |              |          |        |      |
integrativeapproaches.
| brain contribute |        | to this | process.    | To        | accomplish | this    | goal, we |     |     |     |     |     |     |     |     |
| ---------------- | ------ | ------- | ----------- | --------- | ---------- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
| would elect      | to use | an      | exploratory | analysis, |            | such as | the Two- |     |     |     |     |     |     |     |     |
stage or Latent Input Approach. In the Two-stage analysis, we 4.2. Limitationsofusingtheseapproaches
| would simply | fit | our behavioral |     | model | to the | behavioral | data, |     |     |     |     |     |     |     |     |
| ------------ | --- | -------------- | --- | ----- | ------ | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
andcorrelatethe parametersregulatingtheencodingprocess of Inourworkingexampleabove,weidentifiedafewlimitations
our model to say, parameters of the HRF in our neural data. of using various approaches. First, the balancing of fit between
Similarly,intheLatentInputanalysis,wewouldusethetimingof behavioral data, neural data, or both is a key consideration

76 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
in model-based cognitive neuroscience. In general, to optimize &VanZandt,2013a;Turner&Sederberg,2012,2014;Turneretal.,
predictions for say, behavior, it would be better to condition forthcoming;Turner&VanZandt,2012,2014),andasaresult,we
on neural data. However, if one is more interested in the joint maygainnewinsightandinterestinthesenetwork-stylemodels
| distribution | of both | neural | and | behavioral | measures, |     | then the |     |     |     |     |     |     |     |     |
| ------------ | ------- | ------ | --- | ---------- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
inthecomingyears.
| modeling | goals | are more | generative | than | discriminative, |     | and |     |     |     |     |     |     |     |     |
| -------- | ----- | -------- | ---------- | ---- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
conditioning on one variable would introduce limitations. The 4.3. Otherapproaches
authorsofthepresentmanuscripthavedeliberatedbetweenthese
threemodelinggoals,andarrivedatonlyanambiguoussolution:
Althoughtheapproacheswehavepresentedhereencompass
decisionsmustbemadeonacase-by-casebasis,alwayswiththe
themostprevalentapproachestomodel-basedcognitiveneuro-
researcher’sgoalsinmind.
|         |            |     |            |     |             |     |          | science, | other approaches |     | have | been | used to | gain a better | un- |
| ------- | ---------- | --- | ---------- | --- | ----------- | --- | -------- | -------- | ---------------- | --- | ---- | ---- | ------- | ------------- | --- |
| Second, | constraint | is  | not always | a   | good thing. | If  | one does |          |                  |     |      |      |         |               |     |
derstandingofhowthebrainproducesabehavior.Onestructural
nothavestrongintuitionabouthowcomponentsofamodelare
exampleistousesomeexperimentalvariablethathypothetically
| carried out | in the     | brain, | it would | be unwise       | to  | impose    | strong |             |        |                |          |                |          |           |         |
| ----------- | ---------- | ------ | -------- | --------------- | --- | --------- | ------ | ----------- | ------ | -------------- | -------- | -------------- | -------- | --------- | ------- |
|             |            |        |          |                 |     |           |        | affects the | neural | data           | to split | the behavioral |          | data into | differ- |
| constraints | on a       | model. | One way  | of autonomously |     | carrying  | out    |             |        |                |          |                |          |           |         |
|             |            |        |          |                 |     |           |        | ent levels. | Once   | the behavioral |          | data is        | divided, | the data  | can be  |
| justifiable | constraint | is     | to use   | the approaches  |     | discussed | here   |             |        |                |          |                |          |           |         |
fitandevaluatedonthebasisofdifferencesinparametervalues.
| along a continuum |     | of increasingly |     | more | confirmatory |     | research. |     |     |     |     |     |     |     |     |
| ----------------- | --- | --------------- | --- | ---- | ------------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
Asanothertack,onecouldusesomeoftheapproachesdiscussed OneexampleofthisisinParkinson’sDisease,wheredrugtherapy
heretoimposevaryinglevelsofconstraint,moderatingthelevels iscommonlyadministeredtocompensatefordecreasedlevelsof
ofanalysesbetweenexploratoryandconfirmatory.Forexample, dopamine.Frank(2006)makepredictionsforbehavioraldatafor
in the Joint Modeling Approach, one can impose a completely subjectsonandoffmedicationinaGo/NoGotask,andaprobabilis-
ticlearningtask.Theyusedacomputationalneuralnetworkmodel
uninformativepriorontheparametersofthelinkingfunctionand
tomakeconcretepredictionsfordifferencesintaskbehaviorbased
| specify that | all parameters |     | of  | the behavioral | model | be  | mapped |     |     |     |     |     |     |     |     |
| ------------ | -------------- | --- | --- | -------------- | ----- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
onactivationofthesubthalamicnucleus.Frank(2006)foundthat
| totheneural | data.Such |     | ananalysisiswildlyexplorative, |     |     |     | would |     |     |     |     |     |     |     |     |
| ----------- | --------- | --- | ------------------------------ | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
theirmodelaccuratelycapturedthedynamicsofactivityinareas
| be difficult | to implement, |     | and | would convey |     | little information |     |     |     |     |     |     |     |     |     |
| ------------ | ------------- | --- | --- | ------------ | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ofthebasalganglia,andhowthispatternofactivityrelatedtody-
| about the | covariation | between |     | the measures. |     | To move | toward |     |     |     |     |     |     |     |     |
| --------- | ----------- | ------- | --- | ------------- | --- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
namicadjustmentsinresponsethresholds.Asimilarmechanism
| a more confirmatory |     | regime, | one | could | impose | a stronger | prior |     |     |     |     |     |     |     |     |
| ------------------- | --- | ------- | --- | ----- | ------ | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
derivedfromsay,previousresearchorinvestigationoftheprior was later found in impulse control for Parkinson’s patients with
predictivedistribution(Vanpaemel,2010,2011;Vanpaemel&Lee, deepbrainstimulationusingasimilaranalysisdesign(Cavanagh
| 2012).Similarly,onecouldconstrainthesetofparametersthatare |     |     |     |     |     |     |     | etal.,2011). |     |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
relatedtotheneuraldatabysimplysettingelementsofthelinking The examples above illustrate an analytic approach where
functiontozero.Suchananalysiswouldprovideagreatertestof experimental variables guide the analysis of the behavioral data
themodel,butwouldalsoforcethemodeltorelymoreheavilyon on the basis of how those variables affect the neural data.
thejointdistributionofthemeasures. Another type of analysis takes the effects of the neural data
Third, in this article, we have emphasized structural connec- one step further (e.g., Kiani, Hanks, & Shadlen, 2008; Mazurek,
tions that are largely at one level. This is a limitation because Roitman,Ditterich,&Shadlen,2003;Ratcliff,Cherian,&Segraves,
| the behavioral | data  | can          | be thought | of as        | the end | result  | of some |                |       |            |               |           |          |           |        |
| -------------- | ----- | ------------ | ---------- | ------------ | ------- | ------- | ------- | -------------- | ----- | ---------- | ------------- | --------- | -------- | --------- | ------ |
|                |       |              |            |              |         |         |         | 2003; Ratcliff | et    | al., 2011; | Ratcliff,     | Hasegawa, |          | Hasegawa, | Smith, |
| brain process, | again | highlighting |            | the mismatch |         | between | Marr’s  |                |       |            |               |           |          |           |        |
|                |       |              |            |              |         |         |         | & Segraves,    | 2007; | Ratcliff,  | Philiastides, |           | & Sajda, | 2009).    | For    |
(1982)implementationandcomputationallevelsofanalyseswe example,Ratcliffetal.(2009)usedsingle-trialamplitudemeasures
discussedearlier.Anotherapproachwouldbetoimposestructural of EEG activity in a perceptual decision making experiment to
connectionsthataremulti-level,whereamodelusestheimple- divide their behavioral data into separate groups. Next, Ratcliff
mentationleveltodrivesomemechanisms,andthecomputational etal.fittheDDMtothedatafromeachoftheseseparategroupsand
leveltodriveothers.Asahypotheticalexample,theimplementa- usedestimatesofthedriftrateparametertoshowearlycomponent
tionlevelcouldbeusedtodriveanevidenceaccumulationprocess EEG signals were not reflective of the decision process, whereas
thatremainsunaffectedbyexperimentalinstructions(i.e.,compu- latecomponentEEGsignalsshowedapositivecorrelationtothe
tationalgoals),whereasothermechanismssuchasboundarysep- stimulusevidence(i.e.,thedriftrate).Thistypeofanalysisissimilar
arationorbiascouldbecarriedoutbyotherbrainareasthatare
|     |     |     |     |     |     |     |     | to the Latent | Input | Approach, |     | but with | the flow | of information |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ----- | --------- | --- | -------- | -------- | -------------- | --- |
systematicallyadjustedinresponsetotaskdemands.Suchamodel
movingfromtheneuralmeasurestothebehavioralones.Byusing
wouldbridgethelevelsofanalysisinawaythatmightactuallybe
|     |     |     |     |     |     |     |     | the neural | data | to guide | the search | for | differences | in behavioral |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | -------- | ---------- | --- | ----------- | ------------- | --- |
reflectedinthebrain(Frank,2015).
|     |     |     |     |     |     |     |     | model parameters, |     | we  | can better | understand |     | the mechanistic |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | ---------- | ---------- | --- | --------------- | --- |
Finally,theimpositionofstructureneednotarisefromamodel
|              |         |          |     |               |     |            |     | properties | of these | neural | features | by  | interpreting | them | in the |
| ------------ | ------- | -------- | --- | ------------- | --- | ---------- | --- | ---------- | -------- | ------ | -------- | --- | ------------ | ---- | ------ |
| of behavior. | In this | article, | we  | have oriented | the | approaches | to  |            |          |        |          |     |              |      |        |
nativelanguageofthedecisionmodel.
analysisarounddeterminingwheremechanismsinthemodelare
carriedoutinthebrain.However,onecaneasilyimaginereversing
|                 |     |                |     |                |     |     |            | 5. Conclusions |     |     |     |     |     |     |     |
| --------------- | --- | -------------- | --- | -------------- | --- | --- | ---------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| the orientation |     | to determining |     | how structural |     | and | functional |                |     |     |     |     |     |     |     |
differencesinthebrainmanifestbehaviorally.Suchanendeavor
Thefieldofcognitivesciencehasonlybeguntorealizethefull
beginswiththedevelopmentofagenerativemodeloftheneural
potentialofcombiningbrainandbehaviorasawaytostudythe
data,usuallyformedbyobservingtheinterconnectednessofkey
mind.However,thefieldreliesonthevariousapproachesdevel-
brainregions(Cavanaghetal.,2011;Frank,2006;Ratcliff&Frank,
2012;Wong&Wang,2006),andendsinmappingthesystematic oped by different groups of methodological experts. Due to the
activations of these brain areas to a model of the behavioral seeminglydisjointwaystostudycognition,manyneuroscientists
data.Thesemodelscanbedifficulttoimplementandtestinthe andcognitivemodelersareunawareoftheirmodelingoptions,as
traditionalcognitivemodelingway(e.g.,Busemeyer&Diederich, wellasthebenefitsandlimitationsofdifferentapproaches.Inthis
2010;Heathcoteetal.,2015;Lee&Wagenmakers,2013;Shiffrin, article,wehavedescribedthecurrentlyprominentgeneralmeth-
Lee, Kim, & Wagenmakers, 2008), because they rely on many odsforintegratingneuralandbehavioralmeasures,whileprovid-
parametersandcomplexsimulationstovalidatethem.However, ingsomeexamplesoftheiruseincognitiveneuroscience.Wethen
newmethodshavebeendevelopedtobetterelucidatesimulation- attemptedtoorganizetheseapproachesonthebasisofavariety
basedmodels(forapplicationsinpsychology,seeTurner,Dennis, offactors:thenumberofstages,thecommitmenttoaparticular

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 77
theory,thetypeofinformationflow,thedifficultyofimplemen- Dutilh, G., Vandekerckhove, J., Forstmann, B., Keuleers, E., Brysbaert, M., &
tation,andthetypeofexploration.Weconcludedwithadiscus- Wagenmakers, E. (2012). Testing theories of post-error slowing. Attention,
Perception&Psychophysics,74,454–465.
sionoflimitationsandfurtherconsiderationsinapproachingthe
Eichele, T., Debener, S., Calhoun, V. D., Specht, K., Engel, A. K., Hugdahl, K.,
integrationproblem.Ourcomparisonoftheapproaches(seeFig.1, von Cramon, D. Y., & Ullsperger, M. (2008). Prediction of human errors
and Table 1) highlights that a broad spectrum of methods exist bymaladaptivechangesinevent-relatedbrainnetworks.Proceedingsofthe
forperformingmodel-basedcognitiveneuroscience,andthereare NationalAcademyofSciencesoftheUnitedStates,16,6173–6178.
Erlich,J.C.,Brunton,B.W.,Duan,C.A.,Hanks,T.D.,&Brody,C.D.(2015).Distinct
importantconsiderationsandlimitationsofeachapproach.Inthe
effectsofprefrontalandparietalcortexinactivationsonanaccumulationof
end,weconcludethatmodel-basedapproachesincognitiveneuro- evidencetaskintherat.Elife,4.
scienceareextremelyimportant(cf.Forstmannetal.,2016,2011; Forstmann,B.U.,Anwander,A.,Schäfer,A.,Neumann,J.,Brown,S.,Wagenmakers,
E.-J.,Bogacz,R.,&Turner,R.(2010).Cortico-striatalconnectionspredictcontrol
Mulderetal.,2014;Schall,2004;White&Poldrack,2013),andthe
over speed and accuracy in perceptual decision making. Proceedings of the
choiceofanalysisstronglydependsontheresearchgoal.Itseems NationalAcademyofSciences,107,15916–15920.
tousthathavingaclearlyarticulatedanalyticgoalinmindserves Forstmann,B.U.,Dutilh,G.,Brown,S.,Neumann,J.,vonCramon,D.Y.,Ridderinkhof,
astheimpetusforsuccessfulintegrationbetweenneuroscientific K.R.,&Wagenmakers,E.-J.(2008).Striatumandpre-SMAfacilitatedecision-
makingundertimepressure.ProceedingsoftheNationalAcademyofSciences,
measuresandcognitivetheory.
105,17538–17542.
Forstmann,B.U.,Ratcliff,R.,&Wagenmakers,E.-J.(2016).Sequentialsampling
modelsincognitiveneuroscience:Advantages,applications,andextensions.
References
AnnualReviewsinPsychology,67,641–666.
Forstmann,B.U.,Wagenmakers,E.-J.,Eichele,T.,Brown,S.,&Serences,J.T.(2011).
Anderson,J.A.(1977).Neuralmodelswithcognitiveimplications.InD.LaBerge, Reciprocal relations between cognitive neuroscience an formal cognitive
&S.J.Samuels(Eds.),Basicprocessesinreading:Perceptionandcomprehension models:oppositesattract?TrendsinCognitiveSciences,15,272–279.
(pp.27–90).Hillsdale,NJ:Erlbaum. Frank, M. J. (2006). Hold your horses: A dynamic computational role for the
Anderson,J.R.(2007).Howcanthehumanmindoccurinthephysicaluniverse? subthalamicnucleusindecision-making.NeuralNetworks,19,1120–1136.
NewYork,NY:OxfordUniversityPress. Frank,M.J.(2015).Linkingacrosslevelsofcomputationinmodel-basedcognitive
Anderson,J.R.(2012).Trackingproblemsolvingbymultivariatepatternanalysis neuroscience.InB.U.Forstmann,&E.-J.Wagenmakers(Eds.),Anintroduction
andhiddenMarkovmodelalgorithms.Neuropsychologia,50,487–498. tomodel-basedcognitiveneuroscience(pp.159–177).NewYork:Springer.
Anderson,J.R.,Betts,S.,Ferris,J.L.,&Fincham,J.M.(2010).Neuralimagingto Frank,M.,Gagne,C.,Nyhus,E.,Masters,S.,Wiecki,T.V.,Cavanagh,J.F.,&Badre,D.
trackmentalstates.ProceedingsoftheNationalAcademyofSciencesoftheUnited (2015).fmriandeegpredictorsofdynamicdecisionparametersduringhuman
States,107,7018–7023. reinforcementlearning.JournalofNeuroscience,35(2),485–494.
Anderson,J.R.,Byrne,D.,Fincham,J.M.,&Gunn,P.(2008a).Roleofprefrontaland Gao,J.,Tortell,R.,&McClelland,J.L.(2011).Dynamicintegrationofrewardand
parietalcorticesinassociativelearning.CerebralCortex,18,904–914. stimulusinformationinperceptualdecision-making.PLoSONE,6,1–21.
Anderson,J.R.,Fincham,J.M.,Qin,Y.,&Stocco,A.(2008b).Acentralcircuitofthe Gershman,S.J.,Blei,D.M.,Pereira,F.,&Norman,K.A.(2011).Atopographiclatent
mind.TrendsinCognitiveScience,12,136–143. sourcemodelforfMRIdata.Neuroimage,57,89–100.
Behrens,T.,Woolrich,M.,Walton,M.,&Rushworth,M.(2007).Learningthevalue Gläscher,J.P.,&O’Doherty,(2010).Model-basedapproachestoneuroimaging:
ofinformationinanuncertainworld.NatureNeuroscience,10,1214–1221. combining reinforcement learning theory with fMRI data. WIREs Cognitive
Bishop,C.M.,&Lasserre,J.(2007).Generativeordiscriminative?Gettingthebest Science,1,501–510.
ofbothworlds.BayesianStatistics,8,3–24. Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics.
Boehm,U.,VanMaanen,L.,Forstmann,B.,&VanRijn,H.(2014).Trial-by-trial NewYork:WileyPress.
Grossberg, S. (1978). A theory of human memory: Self-organization and
fluctuations in CNV amplitude reflect anticipatory adjustment of response
performanceofsensory-motorcodes,maps,andplans.InR.Rosen,&F.Snell
caution.NeuroImage,96,95–105.
Bogacz,R.,Brown,E.,Moehlis,J.,Holmes,P.,&Cohen,J.D.(2006).Thephysicsof (Eds.),Progressintheoreticalbiology:Vol.5(pp.233–374).NewYork:Academic
optimaldecisionmaking:Aformalanalysisofmodelsofperformanceintwo- Press.
Hampton,A.N.,Bossaerts,P.,&O’Doherty,J.P.(2006).Theroleoftheventromedial
alternativeforcedchoicetasks.PsychologicalReview,362,1655–1670.
prefrontalcortexinabstractstate-basedinferenceduringdecisionmakingin
Bogacz,R.,Usher,M.,Zhang,J.,&McClelland,J.L.(2007).Extendingabiologically
humans.JournalofNeuroscience,8360–8367.
inspired model of choice: Multi-alternatives, nonlinearity and value-based
Hanes,D.P.,&Schall,J.D.(1996).Neuralcontrolofvoluntarymovementinitiation.
multidimensionalchoice.Themeissueonmodelingnaturalactionselection.
Science,274,427–430.
Philosophical Transactions of the Royal Society: B. Biological Sciences, 362, Hanks,T.D.,Kopec,C.D.,Brunton,B.W.,Duan,C.A.,Erlich,J.C.,&Brody,C.D.
1655–1670. (2015).Distinctrelationshipsofparietalandprefrontalcorticestoevidence
Borst,J.P.,&Anderson,J.R.(2013).Usingmodel-basedfunctionalMRItolocate
accumulation.Nature,520(7546),220–223.
workingmemoryupdatesanddeclarativememoryretrievalsinthefronto-
Heathcote,A.,Brown,S.D.,&Wagemakers,E.-J.(2015).Anintroductiontogood
parietalnetwork.ProceedingsoftheNationalAcademyofSciencesoftheUnited
practicesincognitivemodeling.InB.U.Forstmann,&E.-J.Wagenmakers(Eds.),
States,110,1628–1633.
Anintroductiontomodel-basedcognitiveneuroscience(pp.25–48).NewYork:
Borst,J.P.,Nijboer,M.,Taatgen,N.A.,VanRijn,H.,&Anderson,J.R.(2015).Using
Springer.
data-drivenmodel-brainmappingstoconstrainformalmodelsofcognition.
Ho,T.,Brown,S.,vanMaanen,L.,Forstmann,B.U.,Wagenmakers,E.-J.,&Serences,
PLoSONE,10,e0119673.
J.T.(2012).Theoptimalityofsensoryprocessingduringthespeed-accuracy
Borst,J.P.,Taatgen,N.A.,Stocco,A.,&VanRijn,H.(2010a).Theneuralcorrelates
tradeoff.JournalofNeuroscience,32,7992–8003.
of problem states: Testing fMRI predictions of a computational model of
Horwitz,G.D.,&Newsome,W.T.(1999).Separatesignalsfortargetselectionand
multitasking.PLoSONE,5,e12966.
movementspecificationinthesuperiorcolliculus.Science,284,1158–1161.
Borst,J.P.,Taatgen,N.A.,&VanRijn,H.(2010b).Theproblemstate:Acognitive
Horwitz, G. D., & Newsome, W. T. (2001). Target selection for saccadic eye
bottleneck in multitasking. Journal of Experimental Psychology: Learning,
movements prelude activity in the superior colliculus during a direction-
Memory,&Cognition,36,363–382.
discriminationtask.JournalofNeurophysiology,86,2543–2558.
Broadbent,D.(1985).Aquestionoflevels:CommentonMcClellandandRumelhart. Kemp,C.,&Tenenbaum,J.B.(2008).Structuredmodelsofsemanticcognition.
JournalofExperimentalPsychology:General,114,189–192. Commentary on Rogers and McClelland. Behavioral and Brain Sciences, 31,
Brown,S.,&Heathcote,A.(2008).Thesimplestcompletemodelofchoicereaction
717–718.
time:Linearballisticaccumulation.CognitivePsychology,57,153–178. Kiani,R.,Hanks,T.D.,&Shadlen,M.N.(2008).Boundedintegrationinparietal
Brunton, B. W., Botvinick, M. M., & Brody, C. D. (2013). Rats and humans can cortex underlies decisions even when viewing duration is dictated by the
optimallyaccumulateevidencefordecision-making.Science,340(6128),95–98. environment.JournalofNeuroscience,28,3017–3029.
Busemeyer,J.R.,&Diederich,A.(Eds.)(2010).Cognitivemodeling.Sage. Kim,J.N.,&Shadlen,M.N.(1999).Neuralcorrelatesofadecisioninthedorsolateral
Cavanagh,J.F.,Wiecki,T.V.,Cohen,M.X.,Figueroa,C.M.,Samanta,J.,Sherman,S. prefrontalcortexofthemacaque.NatureNeuroscience,2,176–185.
J.,&Frank,M.J.(2011).Subthalamicnucleusstimulationreversesmediofrontal Lee,M.D.,&Wagenmakers,E.-J.(2013).Bayesianmodelingforcognitivescience:A
influenceoverdecisionthreshold.NatureNeuroscience,14,1462–1467. practicalcourse.CambridgeUniversityPress.
Craigmile,P.,Peruggia,M.,&Zandt,T.V.(2010).HierarchicalBayesmodelsfor Love,B.C.(2015).Thealgorithmiclevelisthebridgebetweencomputationand
responsetimedata.Psychometrika,75,613–632. brain.TopicsinCognitiveScience,7.
Davis,T.,Love,B.C.,&Preston,A.R.(2012).Striatalandhippocampalentropyand Mack,M.L.,Preston,A.R.,&Love,B.C.(2013).Decodingthebrain?salgorithmfor
recognitionsignalsincategorylearning:simultaneousprocessesrevealedby categorizationfromitsneuralimplementation.CurrentBiology,23,2023–2027.
model-basedfmri.Journalofexperimentalpsychology.Learning,memory,and Mansfield, E. L., Karayanidis, F., Jamadar, S., Heathcote, A., & Forstmann, B. U.
cognition,38(4),821–839. (2011).Adjustmentsofresponsethresholdduringtaskswitching:amodel-
Ditterich, J. (2010). A comparison between mechanisms of multi-alternative basedfunctionalmagneticresonanceimagingstudy.JournalofNeuroscience,,
perceptualdecisionmaking:Abilitytoexplainhumanbehavior,predictions 31(41),14688–14692.
for neurophysiology, and relationship with decision theory. Frontiers in Marr,D.(1982).Vision:Acomputationalinvestigationintothehumanrepresentation
Neuroscience,4,184. andprocessingofvisualinformation.NewYork:Freeman.

78 B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79
Massaro, D. W. (1988). Some criticisms of connectionist models of human Ratcliff,R.,Philiastides,M.G.,&Sajda,P.(2009).Qualityofevidenceforperceptual
performance.JournalofMemoryandLanguage,27,213–234. decisionmakingisindexedbytrial-to-trialvariabilityoftheEEG.Proceedings
Mazurek,M.E.,Roitman,J.D.,Ditterich,J.,&Shadlen,M.N.(2003).Aroleforneural oftheNationalAcademyofSciencesoftheUnitedStates,106,6539–6544.
integratorsinperceptualdecisionmaking.CerebralCortex,13,1257–1269. Ratcliff,R.,&Smith,P.L.(2004).Acomparisonofsequentialsamplingmodelsfor
McClelland,J.L.,&Rumelhart,D.E.(1981).Aninteractiveactivationmodelof two-choicereactiontime.PsychologicalReview,111,333–367.
context effects in letter perception: Part 1. An account of basic findings. Rescorla, R. A., & Wagner, A. R. (1972). A theory of pavlovian conditioning:
PsychologicalReview,8,375–407. Variationsintheeffectivenessofreinforcementandnonreinforcement.InA.
McClelland,J.,&Rumelhart,D.(1986).Paralleldistributedprocessing:Explorations H.Black,&W.F.Prokasy(Eds.),ClassicalconditioningII:Currentresearchand
inthemicrostructureofcognition.Vol.2:Psychologicalandbiologicalmodels. theory(pp.64–99).NewYork:AppletonCrofts.
Cambridge,MA:MITPress. Roberts,S.,&Pashler,H.(2000).Howpersuasiveisagoodfit?PsychologicalReview,
McCullogh,W.S.,&Pitts,W.(1943).Alogicalcalculusoftheideasimmanentin 107,358–367.
nervousactivity.BulletinofMathematicalBiophysics,5,115–133. Rosenblatt,M.(1961).Principlesofneurodynamics.Washington,DC:SpartanBooks.
Medin,D.L.,&Schaffer,M.M.(1978).Contexttheoryofclassificationlearning. Roxin,A.,&Ledberg,A.(2008).Neurobiologicalmodelsoftwo-choicedecision
PsychologicalReview,85,207–238. makingcanbereducedtoaone-dimensionalnonlineardiffusionequation.PLoS
Minda,J.P.,&Smith,J.D.(2002).Comparingprototype-basedandexemplar-based ComputationalBiology,4,e1000046.
accountsofcategorylearningandattentionalallocation.JournalofExperimental Rumelhart,D.E.(1977).Towardaninteractivemodelofreading.InS.Dornic(Ed.),
Psychology:Learning,Memory,andCognition,28,275–292. Attentionandperformance(pp.573–603).Hillsdale,NJ:Erlbaum.
Minsky,M.L.,&Papert,S.A.(1969).Perceptrons.Cambridge,MA:TheMITPress.
Rumelhart,D.E.,&McClelland,J.L.(1982).Aninteractiveactivationmodelof
Mittner,M.,Boekel,W.,Tucker,A.M.,Turner,B.M.,Heathcote,A.,&Forstmann,
contexteffectsinletterperception:Part2.Thecontextenhancementeffectand
B.U.(2014).Whenthebraintakesabreak:Amodel-basedanalysisofmind
sometestsandextensionsofthemodel.PsychologicalReview,89,60–94.
wandering.JournalofNeuroscience,34,16286–16295.
Rushworth,M.F. S.,Mars,R.B.,&Summerfield,C.(2009).Generalmechanismsfor
Mohammad-Djafari,A.,&Féron,O.(2006).ABayesianapproachtochangepoint
makingdecisions?CurrentOpinioninNeurobiology,19(1),75–83.
analysisofdiscretetimeseries.InternationalJournalsofImagingSystemsand
Schall,J.D.(2004).Onbuildingabridgebetweenbrainandbehavior.AnnualReview
Technology,16,215–221.
ofPsychology,55,23–50.
Montague, P. R., Dolan, R. J., Friston, K. J., & Dayan, P. (2012). Computational
Shadlen,M.N.,&Newsome,W.T.(1996).Motionperception:Seeinganddeciding.
psychiatry.TrendsinCognitiveScience,16,72–80.
ProceedingsoftheNationalAcademyofSciencesoftheUnitedStates,93,628–633.
Mulder, M., van Maanen, L., & Forstmann, B. U. (2014). Perceptual decision
Shadlen,M.N.,&Newsome,W.T.(2001).Neuralbasisofaperceptualdecisionin
neurosciences–Amodel-basedreview.Neuroscience,277,872–884.
Mulder,M.J.,Wagenmakers,E.-J.,Ratcliff,R.,Boekel,W.,&Forstmann,B.U.(2012). theparietalcortex(areaLIP)oftherhesusmonkey.JournalofNeurophysiology,
Biasinthebrain:Adiffusionmodelanalysisofpriorprobabilityandpotential 86,1916–1936.
Shiffrin, R. M., Lee, M. D., Kim, W., & Wagenmakers, E.-J. (2008). A survey of
payoff.JournalofNeuroscience,32,2335–2343.
Myung,I.J.,&Pitt,M.A.(1997).ApplyingOccam’srazorinmodelingcognition:A modelevaluationapproacheswithatutorialonhierarchicalBayesianmethods.
Bayesianapproach.PsychonomicBulletinandReview,4,79–95. CognitiveScience,32,1248–1284.
Nagai,Y.,Critchley,H.D.,Featherstone,E.,Fenwick,P.B.C.,Trimble,M.R.,&Dolan, Shiffrin,R.M.,&Nobel,P.A.(1997).Theartofmodeldevelopmentandtesting.
R.J.(2004).Brainactivityrelatingtothecontingentnegativevariation:anfmri BehaviorResearchMethods,Instruments,&Computers,29,6–14.
investigation.Neuroimage,21(4),1232–1241. Smith,P.L.(2010).FromPoissonshotnoisetotheintegratedOrnstein-Uhlenbeck
Nunez,M.D.,Srinivasan,R.,&Vandekerckhove,J.(2015).Individualdifferencesin process: Neurally-principled models of diffusive evidence accumulation in
attentioninfluenceperceptualdecisionmaking.FrontiersinPsychology,8(18), decision-makingandresponsetime.JournalofMathematicalPsychology,54,
1–13. 266–283.
O’Doherty,J.P.,Dayan,P.,Friston,K.,Critchley,H.,&Dolan,R.J.(2003).Temporal Smith,P.L.,&McKenzie,C.R. L.(2011).Diffusiveinformationaccumulationby
differencemodelsandreward-relatedlearninginthehumanbrain.Neuron,28, minimalrecurrentneuralmodelsofdecision-making.NeuralComputation,23,
329–337. 2000–2031.
O’Doherty,J.P.,Hampton,A.,&Kim,H.(2007).Model-BasedfMRIanditsapplication Smolensky,P.(1988).Onthepropertreatmentofconnectionism.Behavioraland
torewardlearninganddecisionmaking.AnnalsoftheNewYorkAcademyof BrainSciences,11,1–23.
Science,1104,35–53. Summerfield,C.,&Koechlin,E.(2010).Economicvaluebiasesuncertainperceptual
Ossmy,O.,Moran,R.,Pfeffer,T.,Tsetsos,K.,Usher,M.,&Donner,T.H.(2013). choicesintheparietalandprefrontalcortices.FrontiersinHumanNeuroscience,
The timescale of perceptual evidence integration can be adapted to the 4,208.
environment.CurrentBiology,23,981–986. Teller,D.Y.(1984).Linkingpropositions.VisionResearch,24,1233–1246.
Palmeri,T.J.(2014).Anexemplarofmodel-basedcognitiveneuroscience.Trendsin Teodorescu, A. R., & Usher, M. (2013). Disentangling decision models – from
CognitiveScience,18,67–69. independencetocompetition.PsychologicalReview,120,1–38.
Palmeri,T.,Schall,J.,&Logan,G.(2015).Neurocognitivemodellingofperceptual Tsetsos, K., Gao, G., McClelland, J. L., & Usher, M. (2012). Using time-verying
decisions.InJ.R.Busemeyer,J.Townsend,Z.J.Wang,&A.Eidels(Eds.),Oxford evidencetotestmodelsofdecisiondynamics:Boundeddiffusionvs.theleaky
handbook of computational and mathematical psychology. Oxford University competingaccumulatormodel.FrontiersinNeuroscience,6,1–17.
Press. Tsetsos,K.,Usher,M.,&McClelland,J.L.(2011).Testingmulti-alternativedecision
Peruggia,M.,VanZandt,T.,&Chen,M.(2002).WasitacaroracatIsaw?Ananalysis modelswithnon-stationaryevidence.FrontiersinNeuroscience,5,1–18.
ofresponsetimesforwordrecognition.CaseStudiesinBayesianStatisticsVI, Turner, B. M. (2015). Constraining cognitive abstractions through Bayesian
319–334. modeling.InB.U.Forstmann,&E.-J.Wagenmakers(Eds.),Anintroductionto
Pinker,S.,&Prince,A.(1988).Onlanguageandconnectionism:Analysisofaparallel
model-basedcognitiveneuroscience(pp.199–220).NewYork:Springer.
distributedprocessingmodeloflanguageacquisition.Cognition,28,73–193.
Turner,B.M.,Dennis,S.,&VanZandt,T.(2013a).Bayesiananalysisofmemory
Plichta,M.M.,Wolf,I.,Hohmann,S.,Baumeister,S.,Boecker,R.,Schwarz,A.J.,Zangl,
models.PsychologicalReview,120,667–678.
M.,Mier,D.,Diener,C.,Meyer,P.,Holz,N.,Ruf,M.,Gerchen,M.F.,Bernal-Casas,
Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.-J.,Brown,S.D.,Sederberg,P.B.,
D.,Kolev,V.,Yordanova,J.,Flor,H.,Laucht,M.,Banaschewski,T.,Kirsch,P.,
&Steyvers,M.(2013b).Abayesianframeworkforsimultaneouslymodeling
Meyer-Lindenberg,A.,&Brandeis,D.(2013).Simultaneouseegandfmrireveals
neuralandbehavioraldata.NeuroImage,72,193–206.
acausallyconnectedsubcortical-corticalnetworkduringrewardanticipation.
Turner,B.M.,&Sederberg,P.B.(2012).ApproximateBayesiancomputationwith
JournalofNeuroscience,33(36),14526–14533.
differentialevolution.JournalofMathematicalPsychology,56,375–385.
Purcell,B.,Heitz,R.,Cohen,J.,Schall,J.,Logan,G.,&Palmeri,T.(2010).Neurally-
Turner,B.M.,&Sederberg,P.B.(2014).Ageneralized,likelihood-freemethodfor
constrainedmodelingofperceptualdecisionmaking.PsychologicalReview,117,
parameterestimation.PsychonomicBulletinandReview,21,227–250.
1113–1143.
Purcell,B.A.,&Palmeri,T.J.(2017).Relatingaccumulatormodelparametersand Turner, B. M., Sederberg, P. B., Brown, S., & Steyvers, M. (2013c). A method
neuraldynamics.JournalofMathematicalPsychology,76,156–171. for efficiently sampling from distributions with correlated dimensions.
Purcell,B.,Schall,J.,Logan,G.,&Palmeri,T.(2012).Gatedstochasticaccumulator PsychologicalMethods,18,368–384.
modelofvisualsearchdecisionsinFEF.JournalofNeuroscience,32,3433–3446. Turner,B.M.,Sederberg,P.B.,&McClelland,J.L.(2015a).Bayesiananalysisof
Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85,59–108. simulation-basedmodels(forthcoming).
Ratcliff,R.,Cherian,A.,&Segraves,M.(2003).Acomparisonofmacaquebehavior Turner,B.M.,VanMaanen,L.,&Forstmann,B.U.(2015b).Combiningcognitiveab-
andsuperiorcolliculusneuronalactivitytopredictionsfrommodelsofsimple stractionswithneurophysiology:Theneuraldriftdiffusionmodel.Psychological
two-choicedecisions.JournalofNeurophysiology,90,1392–1407. Review,122,312–336.
Ratcliff,R.,&Frank,M.J.(2012).Reinforcement-baseddecisionmakingincorticos- Turner, B. M., & Van Zandt, T. (2012). A tutorial on approximate Bayesian
triatalcircuits:Mutualconstraintsbyneurocomputationalanddiffusionmod- computation.JournalofMathematicalPsychology,56,69–85.
els.NeuralComputation,24,1186–1229. Turner, B. M., & Van Zandt, T. (2014). Hierarchical approximate Bayesian
Ratcliff,R.,Hasegawa,Y.T.,Hasegawa,Y.P.,Childers,R.,Smith,P.L.,&Segraves, computation.Psychometrika,79,185–209.
M. A. (2011). Inhibition in superior colliculus neurons in a brightness Umakantha,A.,Purcell,B.,&Palmeri,T.(2016).Mappingbetweenaspikingneu-
discriminationtask?.NeuralComputation,23,1790–1820. ral network model and the diffusion model of perceptual decision making
Ratcliff,R.,Hasegawa,Y.T.,Hasegawa,Y.P.,Smith,P.L.,&Segraves,M.A.(2007). (workingtitle),manuscriptinpreparation.
Dualdiffusionmodelforsingle-cellrecordingdatafromthesuperiorcolliculus Usher,M.,&McClelland,J.L.(2001).Onthetimecourseofperceptualchoice:The
inabrightness-discriminationtask.JournalofNeurophysiology,97,1756–1774. leakycompetingaccumulatormodel.PsychologicalReview,108,550–592.

B.M.Turneretal./JournalofMathematicalPsychology76(2017)65–79 79
vanMaanen,L.,Brown,S.D.,Eichele,T.,Wagenmakers,E.-J.,Ho,T.,&Serences, Weissman,D.H.,Roberts,K.C.,Visscher,K.M.,&Woldorff,M.G.(2006).Theneural
J.(2011).Neuralcorrelatesoftrial-to-trialfluctuationsinresponsecaution. basesofmomentarylapsesinattention.NatureNeuroscience,9,971–978.
JournalofNeuroscience,31,17488–17495. White,C.N.,Congdon,E.,Mumford,J.A.,Karlsgodt,K.H.,Sabb,F.W.,Freimer,
vanMaanen,L.,&VanRijn,H.(2010).ThelocusoftheGrattoneffectinpicture-word N. B., London, E. D., Cannon, T. D., Bilder, R. M., & Poldrack, R. A. (2014).
interference.TopiCSinCognitiveScience,2,168–180. Decomposing decision components in the stop-signal task: A model-based
van Maanen, L., Van Rijn, H., & Borst, J. P. (2009). Stroop and picture-word approachtoindividualdifferencesininhibitorycontrol.JournalofCognitive
interferencearetwosidesofthesamecoin.PsychonomicBulletinandReview, Neuroscience,26,1601–1614.
16,987–999. White,C.N.,&Poldrack,R.A.(2013).UsingfMRItoconstraintheoriesofcognition.
vanMaanen,L.,VanRijn,H.,&Taatgen,N.A.(2012).RACE/A:Anarchitectural PerspectivesonPsychologicalScience,8,79–83.
account of the interactions between learning, task control, and retrieval Wiecki,T.V.,Poland,J.,&Frank,M.J.(2015).Model-basedcognitiveneuroscience
dynamics.CognitiveScience,36,62–101. approachestocomputationalpsychiatry:Clusteringandclassification.Clinical
Vanpaemel,W.(2010).Priorsensitivityintheorytesting:AnapologiafortheBayes PsychologicalScience,3.
factor.JournalofMathematicalPsychology,54,491–498. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time
Vanpaemel,W.(2011).Constructinginformativemodelpriorsusinghierarchical integrationinperceptualdecisions.TheJournalofNeuroscience,26,1314–1328.
methods.JournalofMathematicalPsychology,55,106–117. Woodman,G.,Kang,M.,Thompson,K.,&Schall,J.(2008).Theeffectofvisualsearch
Vanpaemel,W.,&Lee,M.D.(2012).Usingpriorstoformalizetheory:Optimal efficiencyonresponsepreparation:Neurophysiologicalevidencefordiscrete
attentionandthegeneralizedcontextmodel.PsychonomicBulletinandReview, flow.PsychologicalScience,19,128–136.
19,1047–1056. Zaki,S.R.,Nosofsky,R.M.,Stanton,R.D.,&Cohen,A.(2003).Prototypeandexemplar
van Ravenzwaaij, D., van der Maas, H. L. J., & Wagenmakers, E. J. (2012). accounts of category learning and attentional allocation: A reassessment.
Optimaldecisionmakinginneuralinhibitionmodels.PsychologicalReview,119, Journal of Experimental Psychology: Learning, Memory and Cognition, 29,
201–215. 1160–1173.
Wagenmakers,E.-J.,Farrell,S.,&Ratcliff,R.(2004).Estimationandinterpretation Zandbelt,B.B.,Purcell,B.A.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2015).Response
of 1/fα noise in human cognition. Psychonomic Bulletin and Review, 11, timesfromensemblesofaccumulators.ProceedingsoftheNationalAcademyof
579–615. SciencesoftheUnitedStatesofAmerica,111,2848–2853.