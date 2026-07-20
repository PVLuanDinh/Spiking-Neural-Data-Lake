JournalofMathematicalPsychology76(2017)59–64
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Editorial
Model-based cognitive neuroscience
ThomasJ.Palmeria,∗ ,BradleyC.Loveb,BrandonM.Turnerc
aVanderbiltUniversity,UnitedStates
bUniversityCollegeLondon,UnitedKingdom
cTheOhioStateUniversity,UnitedStates
h i g h l i g h t s
• Mathematicalpsychologyandcognitiveneurosciencecometogetherinapowerfulnewapproachcalledmodel-basedcognitiveneuroscience.
• Thisapproachcanbothinformcognitivemodelingandhelptointerpretneuralmeasures.
• Thisarticleprovidesanintroductiontothefieldofmodel-basedcognitiveneuroscienceandtothearticlescontainedwithinthisspecialissue.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Thisspecialissueexploresthegrowingintersectionbetweenmathematicalpsychologyandcognitive
Availableonline23November2016 neuroscience. Mathematical psychology, and cognitive modeling more generally, has a rich history
of formalizing and testing hypotheses about cognitive mechanisms within a mathematical and
Keywords:
computationallanguage,makingexquisitepredictionsofhowpeopleperceive,learn,remember,and
Cognitivemodeling
decide. Cognitive neuroscience aims to identify neural mechanisms associated with key aspects of
Cognitiveneuroscience
cognitionusingtechniqueslikeneurophysiology,electrophysiology,andstructuralandfunctionalbrain
Model-basedcognitiveneuroscience
imaging.Thesetwocometogetherinapowerfulnewapproachcalledmodel-basedcognitiveneuroscience,
which can both inform cognitive modeling and help to interpret neural measures. Cognitive models
decomposecomplexbehaviorintorepresentationsandprocessesandtheselatentmodelstatescanbe
usedtoexplainthemodulationofbrainstatesunderdifferentexperimentalconditions.Reciprocally,
neuralmeasuresprovidedatathathelpconstraincognitivemodelsandadjudicatebetweencompeting
cognitivemodelsthatmakesimilarpredictionsaboutbehavior.Asexamples,brainmeasuresarerelated
to cognitive model parameters fitted to individual participant data, measures of brain dynamics are
relatedtomeasuresofmodeldynamics,modelparametersareconstrainedbyneuralmeasures,model
parametersormodelstatesareusedinstatisticalanalysesofneuraldata,orneuralandbehavioraldata
areanalyzedjointlywithinahierarchicalmodelingframework.Weprovideanintroductiontothefield
ofmodel-basedcognitiveneuroscienceandtothearticlescontainedwithinthisspecialissue.
©2016ElsevierInc.Allrightsreserved.
Excitingnewsynergiesbetweenmathematicalpsychologyand 1. Whatismodel-basedcognitiveneuroscience?
cognitive neuroscience have emerged. This special issue of the
JournalofMathematicalPsychologyincludesreviews,tutorials,and Alternative approaches to theory in both psychology and
original research papers highlighting this new area of model- neuroscienceoftenbeginbyconsideringMarr’s(1982)classicthree
basedcognitiveneuroscience.Inthisopeningarticle,weoutlinethis levels:Thecomputationallevelconsidersthegoalsoftheorganism
newapproachandintroducethearticlescontainedinthisspecial and the structure of the environment, without considering
issue. mechanism, typified by many Bayesian theories of the mind
(e.g.,Anderson,1990;Oaksford&Chater,2007;Tenenbaum,Kemp,
Griffiths,&Goodman,2011).Thealgorithmiclevelconsiderswhat
∗ representationsandprocessesunderliecognitionandperception,
Correspondenceto:DepartmentofPsychology,301WilsonHall,Vanderbilt
University,Nashville,TN37240,UnitedStates. without considering their biological realization, typified by
E-mailaddress:thomas.j.palmeri@vanderbilt.edu(T.J.Palmeri). many mathematical and computational models of cognition and
http://dx.doi.org/10.1016/j.jmp.2016.10.010
0022-2496/©2016ElsevierInc.Allrightsreserved.

| 60  |     |     |     |     | Editorial/JournalofMathematicalPsychology76(2017)59–64 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
perception (e.g., Busemeyer, Wang, Townsend, & Eidels, 2015; The emergence and growth of model-based cognitive neuro-
Sun, 2008). The implementation level asks how mechanisms are science over the past decade can be attributed to a number of
physicallyrealizedwithinabiologicalsubstrate,namelyneurons convergingforces.Onewastherecognitiononthepartofcogni-
andtheirconnectionsinthebrain,typifiedbyclassicaltheoretical
|     |     |     |     |     |     |     | tive modelers | and | mathematical |     | psychologists |     | interested | in un- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------ | --- | ------------- | --- | ---------- | ------ |
work in neuroscience (e.g., Carnevale & Hines, 2006; Dayan & derstandingthemechanismsthatbraindataissimplyadditional
Abbott,2005). databywhichtoconstrainandcontrastmodels.Responseprob-
WhileMarrenvisionedconnectionsbetweentheselevels,there abilities, response times, confidence ratings and the like are the
oftenhadbeenintellectualanddisciplinarybarrierstoconsider- outcomes of processing. Brain data reflect intermediary states.
ing explanations that crossed levels. This led theorists to work Considering how internal processes predicted by a model relate
traditionallywithinonlyonelevelofanalysis.Notsolongago,a tointernalprocessesmeasuredinthebraincanbreaktheoretical
graduatestudenttrainedinmathematicalpsychologyconsidering stalemates caused by model mimicry. While two different mod-
postdoctoral training in neuroscience might have been about as els making different mechanistic assumptions about representa-
sensibleasconsideringrunningofftojointhecircus.Forsome,the tionsandprocessesmaymakesimilarpredictionsaboutobserved
braincouldwellbemadeoftinkertoysforitsrelevancetounder- behavior, they may well make different predictions about inter-
standinghumancognition.Aswell,notsolongago,fewtrainedin
nalmodelstates,whichcanthenbecomparedwithorconstrained
systemsneurosciencewouldeverconsiderwhetherinsightsfrom
bymeasuredbrainstates(e.g.,Boucher,Palmeri,Logan,&Schall,
cognitiveandmathematicalpsychologymightinformunderstand-
|     |     |     |     |     |     |     | 2007; Mack, | Preston, | &   | Love, 2013; | Palmeri, |     | 2014; | Purcell et al., |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --- | ----------- | -------- | --- | ----- | --------------- |
ingofneuralfunction.Cognitiveconceptualbuildingblockswere
2010;Purcell,Schall,Logan,&Palmeri,2012).
oftenthoughtlittlemorethanfolkpsychology,withphilosophical
|     |     |     |     |     |     |     | Another | force | was the | recognition |     | on the | part | of cognitive |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----- | ------- | ----------- | --- | ------ | ---- | ------------ |
argumentslendingsupporttoastrictreductionistapproachtoun-
|     |     |     |     |     |     |     | and systems | neuroscientists |     | for | the need | for | new approaches | to  |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --------------- | --- | --- | -------- | --- | -------------- | --- |
derstandingthebrain(e.g.,Churchland,1986).1
makingsenseofthegrowingbodyofneuraldatafromfunctional
Earlyattemptstoaddressthisimpassefocusedonconnectionist
|     |     |     |     |     |     |     | brain imaging, |     | electrophysiology, |     | neurophysiology, |     |     | and other |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------------ | --- | ---------------- | --- | --- | --------- |
modelsofcognitionthattookinspirationfromthebrain.Connec- neurosciencetechniques.Correlatingbrainmeasureswithstimuli,
tionistsviewedthebrainasconsistingofsimplecomputingunits conditions,andresponsesprovidesonlyaratherlimitedwindow
(akintoneurons)thatintegratedsignalspassedacrossconnection
|     |     |     |     |     |     |     | on understanding |     | brain | function. | To  | go beyond | merely | mapping |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----- | --------- | --- | --------- | ------ | ------- |
weightsthatwereadjustedbylearningrules.However,thesemod-
|     |     |     |     |     |     |     | out which | brain | areas or | which | neurons | modulate |     | their activity |
| --- | --- | --- | --- | --- | --- | --- | --------- | ----- | -------- | ----- | ------- | -------- | --- | -------------- |
elsrarelymadecontactwiththeimplementationaldetailsofthe
|     |     |     |     |     |     |     | under which | conditions |     | means | to explain | and | understand | what |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | --- | ----- | ---------- | --- | ---------- | ---- |
brain.Inmostcasesthesemodelsservedasexistenceproofsthat
|     |     |     |     |     |     |     | mechanisms | and | computations |     | are engaged |     | within | those brain |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------ | --- | ----------- | --- | ------ | ----------- |
amodelconsistingofmanysimplecomputingelementscouldac-
areasorneurons.Algorithmicandcomputationalmodelsprovide
complishataskinroughlythesamefashionasahuman.Neverthe-
alanguageandabodyofviablehypotheses,aswellasasetoftools,
less,thesemodelswereattemptstobridgelevelsofanalysisand
|     |     |     |     |     |     |     | for explaining |     | and understanding |     | those | neural | mechanisms | and |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ----------------- | --- | ----- | ------ | ---------- | --- |
werechampionedasmorebiologicallyplausiblethancompeting
computations.
modelsatthealgorithmicorcomputationallevels.Unfortunately,
|            |            |              |      |        |             |        | Recognition  |      | has grown | for    | considering |     | the algorithms | and    |
| ---------- | ---------- | ------------ | ---- | ------ | ----------- | ------ | ------------ | ---- | --------- | ------ | ----------- | --- | -------------- | ------ |
| notions of | biological | plausibility | were | rarely | defined nor | evalu- |              |      |           |        |             |     |                |        |
|            |            |              |      |        |             |        | computations | that | underlie  | neural | processing. |     | Carandini      | (2012) |
atedrigorously.Thegapbetweenlevelsofanalysesstubbornlyre-
characterizedanydirectlinkbetweenneuralcircuitsandbehavior
mained.
Model-basedcognitiveneurosciencebreaksthetraditionalbar- as a ‘‘bridge too far’’, and argued that it was necessary to
riersbetweenmodelsandthebrain(e.g.,Forstmann,Wagenmak- theorizeatanintermediatelevelinMarr’shierarchy,considering
ers,Eichele,Brown,&Serences,2011;Forstmann,2015;Palmeri, the algorithms and computations that neural circuits perform.
Schall, Logan, & Townsend, 2015; Smith & Ratcliff, 2004). From Thepurelybottom-upapproachtounderstandingthebrainthat
|     |     |     |     |     |     |     | characterized | the | initial | stages | of the | billion | Euro Human | Brain |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------- | ------ | ------ | ------- | ---------- | ----- |
theperspectiveofcognitiveandmathematicalpsychology,formal
|     |     |     |     |     |     |     | Project | was widely | criticized |     | by cognitive |     | and computational |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | ---------- | --- | ------------ | --- | ----------------- | --- |
modelsexplainbehaviorintermsofrepresentationsandprocesses
neuroscientistsandledtoashake-upofitsleadershipandvision
instantiatedinmathematicsandcomputations,andobservedvari-
|     |     |     |     |     |     |     | (e.g., Enserink |     | & Kupferschmidt, |     | 2014; | Theil, | 2015). | Rather than |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ---------------- | --- | ----- | ------ | ------ | ----------- |
ationinbehavioracrossexperimentalconditionsandindividuals
|     |     |     |     |     |     |     | adopting | a strictly | bottom-up |     | (or top-down) |     | approach, | model- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --------- | --- | ------------- | --- | --------- | ------ |
isexplainedintermsofvariationinmodelparametersandmodel
basedcognitiveneurosciencecanbecharacterizedasaninside-out
| states. Model-based |     | cognitive | neuroscience |     | allows for consider- |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | --------- | ------------ | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
approach(Love,2015),thatmaywellbealeveloftheorizingthat
ationofwhethertheselatentmodelparametersormodelstates
mightberelatedto,orconstrainedby,observedbrainmeasures isjustright(Logan,Yamaguchi,Schall,&Palmeri,2015).
or brain states, over and above whether a model fits or predicts Perhapsthemostpotentforcepropellingmodel-basedcogni-
observedbehavior.Fromtheperspectiveofsystemsandcognitive tiveneuroscienceoverthepastdecadehasbeenitsdemonstrated
neuroscience,akeycomponentofunderstandingneurons,neural success in providing new insight at both the cognitive and neu-
circuits, or brain areas is explaining the computations that they rallevels.Oneespeciallysalientbodyofworkhascenteredaround
perform.Inamodel-basedcognitiveneuroscienceapproach,tothe accumulator models of decision making, a well-known class of
extentthatbrainmeasuresorbrainstatesarepredictedbymodel modelswithalonghistoryincognitivepsychology(e.g.,Ratcliff&
parametersormodelstates,thosemodelsprovideapotentialex- Smith,2004).Thesemodelsassumethatvariabilityinchoiceprob-
planation of brain function, regardless of whether or not those abilityandresponsetimesarisefromvariabilityinthe,oftennoisy,
modelsareimplementedinneuron-likeelements. accumulationofevidencetoresponsethresholds,andvariantsof
|     |     |     |     |     |     |     | these models | have | accounted |     | for decisions |     | in perception, | mem- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | --------- | --- | ------------- | --- | -------------- | ---- |
ory,categorization,andothertasks(e.g.,Bogacz,Brown,Moehlis,
| 1   |     |     |     |     |     |     | Holmes,&Cohen,2006;Brown&Heathcote,2008;Forstmann,Rat- |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
Ofcourse,therewereexceptionstobarriersbetweenthealgorithmiclevel
|     |     |     |     |     |     |     | cliff, & Wagenmakers, |     | 2016; | Nosofsky |     | & Palmeri, | 1997; | Palmeri, |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | ----- | -------- | --- | ---------- | ----- | -------- |
andtheimplementationlevel,toagaincastthisinMarr’sterms.Inthecaseof
relativelylow-levelvisualsensationandperception,therehavelongbeendeep 1997).Asoneofthefirstexamplesofsystemsneurosciencemak-
connectionsbetweentheoreticalworkinvisualpsychophysicsandtheunderlying ingcontactwithcognitivemodeling,whenHanesandSchall(1996)
visualneurophysiologyandneuroanatomy,inpartbecausetherelevantneural wereinterestedinunderstandinghowneuronsinFrontalEyeField
| hardware is | not far | removed | from the source | of visual | stimulation. | And the |     |     |     |     |     |     |     |     |
| ----------- | ------- | ------- | --------------- | --------- | ------------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
(FEF)decidewhereandwhentosaccadeinthevisualfield,they
fieldofcognitiveneuropsychologyhaslongconsideredtheoreticallyhowcasesof
turnedtothecognitivemodelingliteratureforinspirationandin-
braindamageandneurodegenerativeandneurodevelopmentaldisordersinfluence
sight.BasedonthefactthatthedynamicsofcertainFEFneurons
understandingofhumancognition.

Editorial/JournalofMathematicalPsychology76(2017)59–64 61
mirroredthedynamicsinaccumulators,accumulationofevidence completelyconsistentthroughouttheexperiment(anassumption
modelsprovidedalanguagefordescribingthecomputationsthat knownasindependent,identicallydistributed).However,because
theseFEFneuronswereengagedin.Cognitivemodelsprovidedin- there is no mechanism to guide these fluctuations, we cannot
sightintoneuralprocesses.HanesandSchallalsoshowedthatthe appreciateaspectsofthedecisionprocessthatarevitaltoensuring
dynamics of these FEF neurons were more consistent with vari- success on a specific trial. Furthermore, these assumptions are
ableaccumulationtoafixedthresholdthanfixedaccumulationtoa at odds with several findings in neuroscience that implicate
variablethreshold,twocompetingmechanistichypothesesofdeci- the gradual waxing and waning of attention on behavioral
sionmakingthatcanbedifficulttodistinguishbasedonbehavioral performance. In summary of these findings, unique networks of
dataalone(Grice,1968).Neuraldataprovidedinsightintocogni- brain activity arise from separating neural data on the basis of
tivemodels. behavioraldata:an‘‘off-task’’networkgivesrisetopoorbehavioral
Theseinitialinsightsspawnedaconsiderablebodyofresearch performance whereas an ‘‘on-task’’ network gives rise to good
linkingneurophysiologyandcognitivemodelingtounderstandel- behavioralperformance(Mittneretal.,2014;Turner,VanMaanen,
ementary decision making (e.g., Forstmann et al., 2016; Gold & &Forstmann,2015).
Shadlen,2007;Loganetal.,2015;Mazurek,Roitman,Ditterich,& These observations lead Turner et al. (2015) to develop a
Shadlen,2003;Palmerietal.,2015;Ratcliff,Cherian,&Segraves, modelthatblendsneuroscienceandmathematicalpsychologyto
2003;Schall,2001,2004;Smith&Ratcliff,2004;Zandbelt,Purcell, formally ground decision-making models with neurophysiology.
Palmeri,Logan,&Schall,2014).Asoneillustrativeexample,Pur- Theirstrategywastotreattrial-by-trialneuraldata(asmeasured
celletal.(2010,2012)appliedaccumulatormodelstounderstand by fMRI) as information about the trial-to-trial fluctuations
existingdataonthebehaviorandneurophysiologyofsaccadede- in the latent parameters assumed by a standard diffusion
cisionmakingbyawakebehavingprimates(e.g.,Bichot&Schall, model. The model was constructed on the basis of a previously
1999; Cohen, Heitz, Woodman, & Schall,2009). Adopting a clas- developedframeworkforimposingneurophysiologicalconstraints
sicapproachusedinmathematicalpsychology,theyformulateda onbehavioralmodelsacrosssubjects(Turner,2015;Turneretal.,
varietyofalternativemodelsassumingvariousarchitecturalcom- 2013),butextendsthisframeworktoatrial-by-trialbasis.Oncefit
ponentscharacteristicofvariousaccumulatormodelsofdecision todata,themodelwasabletoarticulatehowdisparatenetworks
making,rejectingmodelsthatcouldnotaccountqualitativelyand ofbrainactivitywereassociatedwithorthogonalmechanismsin
quantitativelyforobservedresponseprobabilitiesanddistribution the model, such as pre-stimulus bias and the rate of evidence
ofresponsetimesforsaccades. accumulation. Turner et al. also showed that not only did their
Going beyond a pure mathematical psychology approach of modelprovideanewperspectiveonbothneuralandbehavioral
fittingmodelstobehavioraldata,theyturnedtoneurophysiology datausinggenerativemodelingtechniques,butthemodelcould
in two ways. First, they allowed neurophysiology to constrain alsooutperformastandarddiffusionmodelthatonlyconsidered
key model components. In many, but not all (e.g., Nosofsky & behavioraldatainaleave-one-outcross-validationtest.
Palmeri,1997;Palmeri,1997),applicationsofaccumulatormodels, Model-basedneuroscienceopensuppossibilitiesforcognitive
the rate at which evidence is accumulated, the drift rate, is modelstotakeonsecondlivesasformalneuroscientifictheories.
allowedtobeafreeparameter.Purcelletal.(2010,2012)instead Forexample,LoveandGureckis(2007)proposedatheorylinking
instantiated an hypothesis that a particular class of neurons in aspectsoftheSUSTAINclusteringmodelofhumancategorization
FEF (visually-responsive neurons) represent the evidence that is (Love, Medin, & Gureckis, 2004; Sakamoto & Love, 2004) to
accumulated, replacing the drift rate and other parameters with the functions of prefrontal cortex and the hippocampus. They
recordedneurophysiology.Neurophysiologysignificantlylimited simulatingvariouspopulations,suchasamnesics(Love&Gureckis,
the flexibility of various model architectures to account for 2007),infants(Gureckis&Love,2004),andtheaged(Davis,Love,
observedbehavioraldata. &Maddox,2012),byadjustingmodelparametershypothesizedto
Second,facedwithseveralalternativemodelarchitecturesthat relate to brain regions whose functions vary across populations.
could account equally well for the observed behavioral data, if Withtheadventofmodel-basedneuroscience,exactpredictionsof
they had no other data to turn to, they would have had to thetheoryweretestedandconfirmedwithhealthyyoungadults
appeal to parsimony in selecting a winning model architecture using fMRI (Davis et al., 2012; Davis, Love, & Preston, 2012a;
(see also Boucher et al., 2007; Logan et al., 2015). Purcell et al. Davis,Xue,Love,Preston,&Poldrack,2014;Mack,Preston,&Love,
(2010,2012)insteadturnedtoneurophysiologyasanadditional in press). The analyses revealed a number of phenomena that
datasourceforcontrastingbetweenalternativemodels.Adopting wouldnotbepossibletoobservewithoutthemodel,suchashow
thelinkingproposition(Schall,2004;Teller,1984)thatmovement- theinvolvementofthehippocampuschangesoverlearningtrials,
related neurons in FEF instantiate an accumulation of evidence rampingupforfamiliaritems(relatedtorecognition)atthetime
toathresholdforsaccadedecisions(Hanes&Schall,1996),they of decision and ramping down at the time of feedback as the
comparedthepredicteddynamicsofmodelaccumulationtothe errorsignalabates(Davisetal.,2012).Themodel-basedimaging
observed dynamics of these FEF neurons (see also Purcell & work also confirmed more speculative hypotheses such as that
Palmeri, 2017). Only their gated accumulator model could both prefrontalandhippocampusinteractionswouldbestrongestinthe
accountforthebehavioraldataandpredictthedynamicsofFEF earlystagesofmasteringanewlearningtaskasattentionweights
movement-related neurons. Neurophysiology provided key data areestablished(Macketal.,inpress).
by which to contrast models that otherwise provided the same While we opened this article by contrasting a bottom-up
predictionsofovertbehavior. neuralnetworkapproachwithaninside-out(Love,2015)cognitive
Another approach for avoiding the theoretical stalemate that modeling approach to relating brain and behavior, we want to
canensuewhenfittingcomplexmodelstobehavioraldataalone makeclearthattheapproachesusedinamodel-basedcognitive
is to treat neural data as auxiliary information on which latent neurosciencecanjustaswellbeappliedtoneuralnetworkmodels
modelmechanismsshouldcovary.Modelsliketheclassicdiffusion as to more abstract cognitive models. The SUSTAIN model (Love
model (Ratcliff & Rouder, 1998; Ratcliff & Smith, 2004) have etal.,2004)usedbyDavisetal.(2012,a);Davis,Love,andPreston
threesourcesoftrial-to-trialvariability,assumingfluctuationsin (2012b)discussedaboveisinstantiatedusinganumberofneural
thingslikeresponsebias,therateofevidenceaccumulation,and network building blocks. Yet the relation between SUSTAIN and
perceptualandmotornon-decisiontime.Theassumptionisthat brainimagingdataisnotcementedbyanymappingfromneural-
theseparametersvaryfromonetrialtoanotherinwaysthatare likemodelelementstoneuronsinthebrain,butbytheabilityof

62 Editorial/JournalofMathematicalPsychology76(2017)59–64
patternsofactivityinthemodeltorevealandexplainpatternsof inaniterativefashion,similarinsomewaystoanEMalgorithm.
activity in the brain. Similarly, the overall structure of so-called As illustration, they show how the variability in the parameters
deep learning models of vision (e.g., LeCun, Bengio, & Hinton, ofanaccumulator(sequentialsampling)modelcanberelatedto
2015) are inspired by neural networks and key aspects of the variabilityinneuroimagingdata.
neurophysiology of the primate visual system. But the insights Howattentioninfluencesperceptualdecisionmaking:single-trialEEG
providedbythesemodelsintounderstandingtherepresentation correlatesofdrift-diffusionmodelparameters.Nunez,Vandekerck-
of objects in the brain (Kriegeskorte, 2015; Yamins et al., 2014) hove,andSrinivasan(2017)provideaspecificillustrationofhow
isbasedonhowwellpatternswithinhigh-levelrepresentational variability in neural measures can constrain variability in model
layers of these models predict patterns of brain activity, not parameters. Within a hierarchical Bayesian framework, various
on the neural-like building blocks of these models (Khaligh- formsofadrift-diffusionmodelarefittedtobehavioraldatafrom
Razavi, Henriksson, Kay, & Kriegeskorte, 2017; Khaligh-Razavi & aperceptualdecisionmakingtask,withdifferentmodelformsas-
Kriegeskorte,2014). sumingdifferentmathematicalrelationshipsbetweenmodelpa-
rametersandEEGmeasures.Trial-to-trialmeasuresofcertainkey
2. Overview attention-relatedevokedpotentialsinsimultaneousEEGrecord-
ingscanexplaintrial-to-trialevidenceaccumulationratesandper-
Hereweprovidebriefoutlinesofthepapersthatappearinthis ceptualprocessingtimesinadiffusionmodelfittedtoperceptual
specialissue: decisionmakingbehavior.
Approaches to analysis in model-based cognitive neuroscience. Aconfirmatoryapproachforintegratingneuralandbehavioraldata
Turner,Forstmann,Love,Palmeri,andVanMaanen(2017)provide into a single model. van Ravenzwaaij, Provost, and Brown (2017)
anoverarchingframeworkfordescribingthevaryingapproaches provideanotherillustrationofajointmodelingapproachtomodel-
tomodel-basedcognitiveneurosciencethathaveemergedinthe based cognitive neuroscience. Within a hierarchical Bayesian
literature over the past several years. They organize these ap- frameworktheyusetheLinearBallisticAccumulator(LBA)model
proachesonthebasisofparticulartheoreticalgoals,whichinclude (Brown&Heathcote,2008)toaccountforbehavioraldataduringa
usingneuraldatatoconstrainacognitivemodel,usingacognitive mentalrotationtask,testingdifferenthypotheseslinkingcognitive
modeltopredictneuraldata,andaccountingforbothneuraland model parameters and neural data measured via event-related
behavioraldatasimultaneouslyusingthesamemodel.Accompa- potentials(ERPs).Theyspecificallyinvestigatehowchangesindrift
nyingeachofthesetheoreticalgoals,theyhighlightsomepartic- rateandnon-decisiontimewithmentalrotationanglemightbe
ularlysuccessfulexamples.Theyalsoprovideaconceptualguide constrainedbychangesincertainERPamplitudesmeasuredduring
tochoosingamongvariousapproacheswhenperformingmodel- thetask.
basedcognitiveneuroscience.
On the efficiency of neurally-informed cognitive models to identify
Integratingtheoreticalmodelswithfunctionalneuroimaging.Pratte latentcognitivestates.Hawkins,Mittner,Forstmann,andHeathcote
and Tong (2017) highlight a number of salient examples linking (2017) illustrate how neural data can be used to test between
cognitivemodelsandfunctionalbrainimagingdatausingamodel- cognitive models with different latent states. They focus on
based cognitive neuroscience approach. Their selective review whethertheunderlyingstatesdrivingperformanceinaspeeded
spans a broad range of core topics in perception and cognition, decisiontasksarediscreteorcontinuous.Throughmodelrecovery
including visual perception (Brouwer & Heeger, 2011), attention studiestheauthorsdeterminethatdiscretestatemodelsaremore
(Pratte,Ling,Swisher,&Tong,2013),long-termmemory(Kragel, robustlyrecoveredthancontinuousstatemodels,suggestingthat
Morton, & Polyn, 2015), categorization (Mack et al., 2013), and neural data may more easily be linked to certain varieties of
cognitivecontrol(Ide,Shenoy,Yu,&Li,2013). cognitivemodels.
A step-by-step tutorial on using the cognitive architecture ACT- Relatingaccumulatormodelparametersandneuraldynamics.Purcell
R in combination with fMRI data. Borst and Anderson (2017) andPalmeri(2017)buildontheworkcitedearlierontheidentifi-
provide a tutorial on using the ACT-R cognitive architecture cationofneuralactivityincertainbrainareaswithevidenceac-
(e.g.,Anderson,Bothell,Lebiere,&Matessa,1998)tounderstand cumulationinsequentialsamplingmodels.Throughsimulations,
fMRIdata(e.g.,Anderson,Betts,Ferris,&Fincham,2010;Anderson, they caution against simply equating variability in measures of
Fincham, Qin, & Stocco, 2008; Borst & Anderson, 2013). They neural dynamics with variability in cognitive model parameters.
illustratehowACT-RcanbeusedincombinationwithfMRIdata Simulatedvariationinmodeldynamicsinaccumulatorsisnotal-
intwodifferentways:firstthatfMRIdatacanbeusedtoevaluate waysrelatedone-to-onewithvariationofaccumulatormodelpa-
and constrain models in ACT-R by means of predefined Region- rameters. The most general mapping between neural measures
of-Interest(ROI)analysis,andsecondthatpredictionsfromACT-R andmodelmechanismsmaybeonebetweenmeasuredneuraldy-
modelscanbeusedtolocateneuralcorrelatesofmodelprocesses namicsandpredictedmodeldynamics,notonebetweenmeasured
and representations by means of model-based fMRI analysis. As neuraldynamicsandmodelparameters.
atutorial,theyprovidecodeandworkedexamplesofbothtypes
A primer on encoding models in sensory neuroscience. van Gerven
ofanalysisonamathproblemsolvingtaskperformedinanfMRI
(2017) explores fundamental questions of how the primate
scanner.
visualsystemrepresentsthevisualworld.Invisualneuroscience,
Variability in behavior that cognitive models do not explain can be the concept of the receptive field has been a key concept
linked to neuroimaging data. Gluth and Rieskamp (2017) review for understanding the response properties of neurons. While
evidence for the proposal that neural and behavioral variability classical receptive field mapping has proved successful for
can be linked to one another by allowing moment-to-moment understandingrepresentationsinearlyvisualareaslikeareaV1,
fluctuationsinneuralmeasures,likefMRIandEEG,toinformtrial- moregeneralmethodsareneededforunderstandinghigher-level
by-trialvariabilityincognitivemodelparameters.Oneapproachto visual representations and to allow for non-invasive mapping
linkingsingle-trialmeasuresofthebraintosingle-trialparameters of visual representations in humans. van Gerven provides a
inmodelshasbeentosimplyregressthemontooneanother.Gluth mathematical and computational primer on Encoding Models,
andRieskampprovideatutorialofanovelandefficientalternative whichatfirstapproximationcanbedescribedasgeneralizationof
approachthatgoesbeyondarawtwo-stagecorrelationalapproach classicalreceptivefieldandpopulationreceptivefieldapproaches,
byincreasingtheresolutionofthesingle-trialparameterestimates allowing for the nonlinear response properties of complex

Editorial/JournalofMathematicalPsychology76(2017)59–64 63
representationsinhigh-levelvisualareasandtheirmanifestation References
infunctionalbrainimagingtobewellcharacterized.
Fixed versus mixed RSA: Explaining visual representations by fixed Anderson,J.R.(1990).Theadaptivecharacterofthought.PsychologyPress.
Anderson,J.R.,Betts,S.,Ferris,J.L.,&Fincham,J.M.(2010).Neuralimagingto
and mixed feature sets from shallow and deep computational
trackmentalstates.ProceedingsoftheNationalAcademyofSciencesoftheUnited
models. Khaligh-Razavi et al. (2017) provide a complementary States,107,7018–7023.
approach to understanding how the primate visual system Anderson,J.R.,Bothell,D.,Lebiere,C.,&Matessa,M.(1998).Anintegratedtheory
represents the world. Their starting point is existing neural oflistmemory.JournalofMemoryandLanguage,38,341–380.
Anderson,J.R.,Fincham,J.M.,Qin,Y.,&Stocco,A.(2008).Acentralcircuitofthe
networkandcomputervisionmodelsofobjectrecognition.Their
mind.TrendsinCognitiveScience,12,136–143.
question is whether the representations produced in the model Bichot, N. P., & Schall, J. D. (1999). Effects of similarity and history on neural
predicttheactivitymeasuredinthebrain.Usingatechniquecalled mechanismsofvisualselection.NatureNeuroscience,2,549–554.
Bogacz,R.(2017).Atutorialonthefree-energyframeworkformodellingperception
Representational Similarity Analysis (Kriegeskorte, 2009, 2015),
andlearning.JournalofMathematicalPsychology,76,198–211.
theyaskwhetherpatternsofsimilaritiesinobjectrepresentations Bogacz,R.,Brown,E.,Moehlis,J.,Holmes,P.,&Cohen,J.D.(2006).Thephysicsof
producedinparticularlayersofamodelareanalogoustopatterns optimaldecisionmaking:Aformalanalysisofmodelsofperformanceintwo-
of similarities measured in particular areas of the brain. Deep alternativeforced-choicetasks.PsychologicalReview,113,700–765.
Borst,J.P.,&Anderson,J.R.(2013).Usingmodel-basedfunctionalMRItolocate
learningmodels(LeCunetal.,2015)haveprovidedgoodaccounts
workingmemoryupdatesanddeclarativememoryretrievalsinthefronto-
of object representations observed in high-level visual areas parietalnetwork.ProceedingsoftheNationalAcademyofSciencesoftheUnited
(Khaligh-Razavi&Kriegeskorte,2014;Kriegeskorte,2015;Yamins States,110,1628–1633.
etal.,2014);thisarticlereviewsthatworkandoutlinesapproaches Borst,J.P.,&Anderson,J.R.(2017).Astep-by-steptutorialonusingthecognitive
architectureACT-RincombinationwithfMRIData.JournalofMathematical
tofixingormixingthemodelrepresentationswhencomparingto
Psychology,76,94–103.
brainmeasures. Boucher,L.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2007).Inhibitorycontrol
Atutorialonthefree-energyframeworkformodelingperceptionand inmindandbrain:Aninteractiveracemodelofcountermandingsaccades.
PsychologicalReview,114,376–397.
learning. Bogacz (2017) provides a tutorial on free energy and
Brouwer,G.J.,&Heeger,D.J.(2011).Cross-orientationsuppressioninhumanvisual
relatedpredictivecodingapproaches.Intheseapproaches,models cortex.JournalofNeurophysiology,106(5),2108–2119.
assume that the sensory cortex infers the most likely values of Brown,S.,&Heathcote,A.(2008).Thesimplestcompletemodelofchoicere-action
attributes or features of sensory stimuli from the noisy inputs time:Linearballisticaccumulation.CognitivePsychology,57,153–178.
Busemeyer,J.R.,Wang,Z.,Townsend,J.T.,&Eidels,A.(2015).TheOxfordhandbook
encoding the stimuli. The author demonstrates how powerful
ofcomputationalandmathematicalpsychology.OxfordUniversityPress.
inferencescanbemadebyverysimplecomputationsthatcouldbe Buss,A.T.,Wifall,T.,Hazeltine,E.,&Spencer,J.P.(2009).Integratingthebehavioral
carriedoutbyneurons.Clearexampleshelpthereadergraspthese andneuraldynamicsofresponseselectioninadual-taskparadigm:Adynamic
general concepts that link measures of uncertainty with neural neuralfieldmodelofDuxetal.(2009).JournalofCognitiveNeuroscience,26(2),
334–351.
computations. Carandini, M. (2012). From circuits to behavior: a bridge too far? Nature
Model-basedfunctionalneuroimagingusingdynamicneuralfields:An Neuroscience,15,507–509.
Carnevale,N.T.,&Hines,M.L.(2006).TheNEURONbook.Cambridge,UK:Cambridge
integrativecognitiveneuroscienceapproach.Wijeakumar,Ambrose,
UniversityPress.
Spencer, and Curtu (2017) provide a review and tutorial of Churchland,P.S.(1986).Neurophilosophy:towardaunifiedscienceofthemind/brain.
an approach to model-based cognitive neuroscience using a Cambridge,MA:MITPress.
theoretical framework called Dynamic Field Theory (Erlhagen & Cohen,J.Y.,Heitz,R.P.,Woodman,G.F.,&Schall,J.D.(2009).Neuralbasisoftheset-
sizeeffectinfrontaleyefield:Timingofattentionduringvisualsearch.Journal
Schöner,2002)appliedtofunctionalbrainimaging(Buss,Wifall, ofNeurophysiology,101,1699–1704.
Hazeltine,&Spencer,2009).TheyoutlinetheassumptionsofDFT Davis,T.,Love,B.C.,&Maddox,W.T.(2012).Age-relateddeclinesinthefidelityof
andhowitisappliedtobehavioraldata,describehowparameters newlyacquiredcategoryrepresentations.LearningandMemory,19,325–329.
Davis,T.,Love,B.C.,&Preston,A.R.(2012a).Learningtheexceptiontotherule:
ofthemodelcanbeusedinbrainimaginganalyses,andcompare
Model-basedfMRIrevealsspecializedrepresentationsforsurprisingcategory
the model-based cognitive neuroscience approach to standard members.CerebralCortex,22,260–273.
brainimaginganalysesofthesamedataset. Davis,T.,Love,B.C.,&Preston,A.R.(2012b).Striatalandhippocampalentropyand
recognitionsignalsincategorylearning:Simultaneousprocessesrevealedby
model-basedfMRI.JournalofExperimentalPsychology:Learning,Memory,and
3. Guestconsultingeditors
Cognition,38,821–839.
Davis, T., Xue, G., Love, B. C., Preston, A. R., & Poldrack, R. A. (2014). Global
We thank the ad hoc reviewers who contributed critical neuralpatternsimilarityasacommonbasisforcategorizationandrecognition
commentsthathelpedshapethepapersappearinginthisspecial memory.JournalofNeuroscience,34(22),7472–7484.
Dayan, P., & Abbott, L. F. (2005). Theoretical neuroscience: Computational and
issue; we especially thank the following for serving as guest
mathematicalmodelingofneuralsystems.Cambridge,MA:MITPress.
consultingeditorsforthisspecialissue:JelmerBorst(Universityof Erlhagen,W.,&Schöner,G.(2002).Dynamicfieldtheoryofmovementpreparation.
Groningen),TylerDavis(TexasTechUniversity),BirteForstmann PsychologicalReview,109(3),545–572.
(UniversityofAmsterdam),ScottBrown(UniversityofNewcastle), Enserink,M.,&Kupferschmidt,K.(2014).Europeanneuroscientistsrevoltagainst
theE.U.’sHumanBrainProject.Science.
Sam Gershman (Harvard University), Laurence Hunt (University
Forstmann,B.U.,Ratcliff,R.,&Wagenmakers,E.J.(2016).Sequentialsampling
College London), Xiaosi Gu (University of Texas at Dallas), modelsincognitiveneuroscience:Advantages,applications,andextensions.
Michael Mack (University of Toronto), Neal Morton (University AnnualReviewofPsychology,67,641–666.
of Texas at Austin), Braden Purcell (New York University), Forstmann,B.U.,Wagenmakers,E.-J.,Eichele,T.,Brown,S.,&Serences,J.T.(2011).
Reciprocal relations between cognitive neuroscience and formal cognitive
MichaelPratte(MississippiStateUniversity),PerSederberg(The
models:Oppositesattract?TrendsinCognitiveSciences,15,272–279.
Ohio State University), Mark Steyvers (University of California Forstmann,B.U.,&Wagenmakers,E.-J.(Eds.)(2015).Anintroductiontomodel-based
Irvine),MarcelvanGerven(DondersInstitute),MariekevanVugt cognitiveneuroscience.Springer.
(University of Groningen), Joachim Vandekerckhove (University Gluth,S.,&Rieskamp,J.(2017).Variabilityinbehaviorthatcognitivemodelsdonot
explaincanbelinkedtoneuroimagingdata.JournalofMathematicalPsychology,
of California Irvine), Corey White (Syracuse University), Bram
76,104–116.
Zandbelt(DondersInstitute). Gold,J.I.,&Shadlen,M.N.(2007).Theneuralbasisofdecisionmaking.Annual
ReviewofNeuroscience,30,535–574.
Grice,G.R.(1968).Stimulusintensityandresponseevocation.PsychologicalReview,
Acknowledgments
75(5),359–373.
Gureckis,T.M.,&Love,B.C.(2004).Commonmechanismsininfantandadult
TJP was supported by NIH R01-EY021833, NSF Temporal Dy- categorylearning.Infancy,5,173–198.
namicsofLearningCenterSMA-1041755,andNIHP30-EY08126; Hanes,D.P.,&Schall,J.D.(1996).Neuralcontrolofvoluntarymovementinitiation.
Science,274,427–430.
BCL was supported by Leverhulme Trust grant RPG-2014-075,
Hawkins, G. E., Mittner, M., Forstmann, B. U., & Heathcote, A. (2017). On the
NIH 1P01-HD080679, and a Wellcome Trust Investigator Award efficiencyofneurally-informedcognitivemodelstoidentifylatentcognitive
WT106931MA. states.JournalofMathematicalPsychology,76,142–155.

64 Editorial/JournalofMathematicalPsychology76(2017)59–64
Ide,J.S.,Shenoy,P.,Yu,A.J.,&Li,C.S.(2013).Bayesianpredictionandevaluationin Pratte, M. S., & Tong, F. (2017). Integrating theoretical models with functional
theanteriorcingulatecortex.JournalofNeuroscience,33(5),2039–2047. neuroimaging.JournalofMathematicalPsychology,76,80–93.
Khaligh-Razavi, S.-M., Henriksson, L., Kay, K., & Kriegeskorte, N. (2017). Fixed Purcell,B.A.,Heitz,R.P.,Cohen,J.Y.,Schall,J.D.,Logan,G.D.,&Palmeri,T.J.(2010).
versusmixedRSA:Explainingvisualrepresentationsbyfixedandmixedfeature Neurally-constrainedmodelingofperceptualdecisionmaking.Psychological
setsfromshallowanddeepcomputationalmodels.JournalofMathematical Review,117,1113–1143.
Psychology,76,184–197. Purcell,B.A.,&Palmeri,T.J.(2017).Relatingaccumulatormodelparametersand
Khaligh-Razavi,S.-M.,&Kriegeskorte,N.(2014).Deepsupervised,butnotunsuper- neuraldynamics.JournalofMathematicalPsychology,76,156–171.
vised,modelsmayexplainITcorticalrepresentation.PLoSComputationalBiol- Purcell,B.A.,Schall,J.D.,Logan,G.D.,&Palmeri,T.J.(2012).Fromsalienceto
ogy,10,e1003915. saccades:Multiple-alternativegatedstochasticaccumulatormodelofvisual
Kragel,J.E.,Morton,N.W.,&Polyn,S.M.(2015).Neuralactivityinthemedial search.JournalofNeuroscience,32(10),3433–3446.
temporalloberevealsthefidelityofmentaltimetravel.JournalofNeuroscience, Ratcliff,R.,Cherian,A.,&Segraves,M.(2003).Acomparisonofmacaquebehavior
35(7),2914–2926. andsuperiorcolliculusneuronalactivitytopredictionsfrommodelsofsimple
Kriegeskorte, N. (2009). Relating population-code representations between two-choicedecisions.JournalofNeurophysiology,90,1392–1407.
man, monkey, and computational models. Frontiers in Neuroscience, 3, Ratcliff,R.,&Rouder,J.N.(1998).Modelingresponsetimesfortwo-choicedecisions.
363–373. PsychologicalScience,9,347–356.
Kriegeskorte,N.(2015).Deepneuralnetworks:Anewframeworkformodeling Ratcliff,R.,&Smith,P.L.(2004).Acomparisonofsequentialsamplingmodelsfor
biologicalvisionandbraininformationprocessing.AnnualReviewofVision two-choicereactiontime.PsychologicalReview,111,333–367.
Science,1,417–446. Sakamoto,Y.,&Love,B.C.(2004).Schematicinfluencesoncategorylearningand
LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), recognitionmemory.JournalofExperimentalPsychology:General,133,534–553.
436–444. Schall,J.D.(2001).Neuralbasisofdeciding,choosingandacting.NatureReviews
Logan,G.D.,Yamaguchi,M.,Schall,G.D.,&Palmeri,T.J.(2015).Inhibitorycontrol Neuroscience,2,33–42.
inmindandbrain2.0:Ablocked-inputmodelofsaccadiccountermanding. Schall,J.D.(2004).Onbuildingabridgebetweenbrainandbehavior.AnnualReview
PsychologicalReview,122,115–147. ofPsychology,55,23–50.
Love,B.C.(2015).Thealgorithmiclevelisthebridgebetweencomputationand Smith,P.L.,&Ratcliff,R.(2004).Psychologyandneurobiologyofsimpledecisions.
brain.TopicsinCognitiveScience,7(2),230–242. TrendsinNeuroscience,27,161–168.
Love,B.C.,&Gureckis,T.M.(2007).Modelsinsearchofabrain.Cognitive,Affective, Sun, R. (Ed.) (2008). The cambridge handbook of computational psychology.
&BehavioralNeuroscience,90–108. CambridgeUniversityPress.
Love,B.C.,Medin,D.L.,&Gureckis,T.M.(2004).SUSTAIN:Anetworkmodelof Teller,D.Y.(1984).Linkingpropositions.VisionResearch,24,1233–1246.
categorylearning.PsychologicalReview,111(2),309–331. Tenenbaum,J.B.,Kemp,C.,Griffiths,T.L.,&Goodman,N.D.(2011).Howtogrowa
Mack,M.L.,Preston,A.R.,&Love,B.C.(2013).Decodingthebrain’salgorithmfor mind:Statistics,structure,andabstraction.Science,331(6022),1279–1285.
categorizationfromitsneuralimplementation.CurrentBiology,23,2023–2027. Theil,S.(2015).Whythehumanbrainprojectwentwrong—andhowtofixit.
Mack,M.L.,Preston,A.R.,&Love,B.C.(2016).Dynamicupdatingofhippocampal ScientificAmerican,313.
objectrepresentationsreflectsnewconceptualknowledge.Proceedingsofthe Turner, B. M. (2015). Constraining cognitive abstractions through Bayesian
NationalAcademyofSciences(inpress). modeling.InB.U.Forstmann,&E.-J.Wagenmakers(Eds.),Anintroductionto
Marr,D.(1982).Vision:Acomputationalinvestigationintothehumanrepresentation model-basedcognitiveneuroscience(pp.199–220).NewYork:Springer.
andprocessingofvisualinformation.NewYork:Freeman. Turner,B.M.,Forstmann,B.U.,Love,B.,Palmeri,T.J.,&VanMaanen,L.(2017).
Mazurek,M.E.,Roitman,J.D.,Ditterich,J.,&Shadlen,M.N.(2003).Aroleforneural Approaches to analysis in model-based cognitive neuroscience. Journal of
integratorsinperceptualdecisionmaking.CerebralCortex,13,1257–1269. MathematicalPsychology,76,65–79.
Mittner,M.,Boekel,W.,Tucker,A.M.,Turner,B.M.,Heathcote,A.,&Forstmann, Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.-J.,Brown,S.D.,Sederberg,P.B.,&
B.U.(2014).Whenthebraintakesabreak:Amodel-basedanalysisofmind Steyvers,M.(2013).ABayesianframeworkforsimultaneouslymodelingneural
wandering.JournalofNeuroscience,34,16286–16295. andbehavioraldata.NeuroImage,72,193–206.
Nosofsky,R.M.,&Palmeri,T.J.(1997).Anexemplar-basedrandomwalkmodelof Turner,B.M.,VanMaanen,L.,&Forstmann,B.U.(2015).Combiningcognitiveab-
speededclassification.PsychologicalReview,104,266–299. stractionswithneurophysiology:Theneuraldriftdiffusionmodel.Psychological
Nunez,M.D.,Vandekerckhove,J.,&Srinivasan,R.(2017).Howattentioninfluences Review,122,312–336.
perceptualdecisionmaking:Single-trialEEGcorrelatesofdrift-diffusionmodel vanGerven,M.A.J.(2017).Aprimeronencodingmodelsinsensoryneuroscience.
parameters.JournalofMathematicalPsychology,76,117–130. JournalofMathematicalPsychology,76,172–183.
Oaksford,M.,&Chater,N.(2007).Bayesianrationality:Theprobabilisticapproachto vanRavenzwaaij,D.,Provost,A.,&Brown,S.D.(2017).Aconfirmatoryapproach
humanreasoning.OxfordUniversityPress. for integrating neural and behavioral data into a single model. Journal of
Palmeri,T.J.(1997).Exemplarsimilarityandthedevelopmentofautomaticity. MathematicalPsychology,76,131–141.
Journal of Experimental Psychology: Learning, Memory, and Cognition, 23, Wijeakumar, S., Ambrose, J. P., Spencer, J. P., & Curtu, R. (2017). Model-based
324–354. functionalneuroimagingusingdynamicneuralfields:Anintegrativecognitive
Palmeri,T.J.(2014).Anexemplarofmodel-basedcognitiveneuroscience.Trendsin neuroscienceapproach.JournalofMathematicalPsychology,76,212–235.
CognitiveScience,18(2),67–69. Yamins,D.L.,Hong,H.,Cadieu,C.F.,Solomon,E.A.,Seibert,D.,&DiCarlo,J.J.
Palmeri, T. J., Schall, J. D., & Logan, G. D. (2015). Neurocognitive modeling of (2014).Performance-optimizedhierarchicalmodelspredictneuralresponses
perceptualdecisionmaking.InJ.R.Busemeyer,J.Townsend,Z.J.Wang,&A. inhighervisualcortex.ProceedingsoftheNationalAcademyofSciences,111(23),
Eidels(Eds.),Oxfordhandbookofcomputationalandmathematicalpsychology. 8619–8624.
OxfordUniversityPress. Zandbelt,B.B.,Purcell,B.A.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2014).Response
Pratte,M.S.,Ling,S.,Swisher,J.D.,&Tong,F.(2013).Howattentionextractsobjects timesfromensemblesofaccumulators.ProceedingsoftheNationalAcademyof
fromnoise.JournalofNeurophysiology,100(6),1346–1356. Sciences,111(7),2848–2853.