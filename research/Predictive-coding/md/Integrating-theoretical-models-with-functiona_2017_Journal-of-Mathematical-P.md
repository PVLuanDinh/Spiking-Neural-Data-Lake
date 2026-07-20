JournalofMathematicalPsychology76(2017)80–93
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Integrating theoretical models with functional neuroimaging
MichaelS.Prattea,b,∗ ,FrankTongb
aDepartmentofPsychology,MississippiStateUniversity,UnitedStates
bDepartmentofPsychologyandtheVanderbiltVisionResearchCenter,VanderbiltUniversity,UnitedStates
a r t i c l e i n f o a b s t r a c t
Articlehistory: Thedevelopmentofmathematicalmodelstocharacterizeperceptualandcognitiveprocessesdatesback
Availableonline25July2016 almost to the inception of the field of psychology. Since the 1990s, human functional neuroimaging
has provided for rapid empirical and theoretical advances across a variety of domains in cognitive
neuroscience.Inmorerecentwork,formalmodelingandneuroimagingapproachesarebeingsuccessfully
combined,oftenproducingmodelswithalevelofspecificityandrigorthatwouldnothavebeenpossible
by studying behavior alone. In this review, we highlight examples of recent studies that utilize this
combinedapproachtoprovidenovelinsightsintothemechanismsunderlyinghumancognition.The
studies described here span domains of perception, attention, memory, categorization, and cognitive
control,employingavarietyofanalyticandmodel-inspiredapproaches.Acrossthesediversestudies,a
commonthemeisthatindividuallytailored,creativesolutionsareoftenneededtoestablishcompelling
links between multi-parameter models and complex sets of neural data. We conclude that future
developmentsinmodel-basedcognitiveneurosciencewillhavegreatpotentialtoadvanceourtheoretical
understandingandabilitytomodelbothlow-levelandhigh-levelcognitiveprocesses.
©2016ElsevierInc.Allrightsreserved.
1. Introduction afieldthatisoftencreditedasoriginatingfromMarr’swork.Inthis
review, we consider a newly emerging endeavor that has arisen
The general goal of cognitive psychology has been to under- fromthemergerofcognitivepsychology,theoreticalmodelingand
standpsychologicalprocessesatwhatMarr(1982)wouldcallthe neuroscience:usingtheoreticalmodelsinconjunctionwithhuman
algorithmic or representational level (see Love, 2015). In order to neuroimagingtostudypsychologicalprocesses.
explorethealgorithmsandrepresentationalstructuresthatmight Advancesinmathematicalandcomputationalapproacheshave
underlieprocessessuchasattentionormemory,cognitivepsychol- played a key role in fMRI since its invention. Some of these de-
ogistsoftenproposeformaltheoreticalmodelsoftheseprocesses, velopmentsincludeanalyticalapproachesforextractingrelevant
and test them by assessing predicted patterns in behavior. For information from the BOLD signal across the temporal domain,
decades,doingsohasprovidedremarkableinsightsintohowthe suchastheuseoftemporalphase-encodeddesigns(Engel,2012;
mindworks.Duringthesametime,thefieldofneurosciencehas Engel et al., 1994; Sereno et al., 1995) and the development of
madestridesinunderstandingwhatMarrcallstheimplementation de-convolution approaches for fast-event related designs (Boyn-
level,or,howtheseprocessesareimplementedinthebiological ton, Engel, Glover, & Heeger, 1996; Buckner et al., 1996; Glover,
machinerythatmakesupthebrain.Morerecently,themergerof 1999).Developmentsininferentialstatisticaltechniqueshavepro-
thesefieldsintoaunifiedcognitiveneurosciencehasresultedinpart duced a number of tools that have helped make fMRI mapping
fromthedevelopmentofnewneuroimagingtechniques,suchas studiessosuccessful,especiallyregardingtechniquestoaccount
functionalmagneticresonanceimaging(fMRI),whichhavemade forwhatisarguablythemostseriousmultiplecomparisonsprob-
investigatingthebiologicalsubstratesofhumancognitionpossi- leminpsychology(seeNichols&Hayasaka,2003).Giventhemul-
ble.Themoretargetedapproachofcombiningtheoreticalmodel- tivariatenatureoffMRIdata,correlation-basedapproaches(Haxby
ingandneurosciencehasbeentermedcomputationalneuroscience, etal.,2001),machinelearningtechniques(Kamitani&Tong,2005;
Norman, Polyn, Detre, & Haxby, 2006; Tong & Pratte, 2012) and
voxel-basedmodelingapproaches(Brouwer&Heeger,2009;Kay,
∗ Naselaris,Prenger,&Gallant,2008;Serences&Saproo,2009)have
Correspondenceto:MississippiStateUniversity,DepartmentofPsychology,
255LeeBlvd,Starkville,MS39762,UnitedStates. beenusedtocapturethecomplexitiesofthesehigh-dimensional
E-mailaddress:prattems@gmail.com(M.S.Pratte). datasets,providingpowerfulnewwaysofidentifyingperceptual
http://dx.doi.org/10.1016/j.jmp.2016.06.008
0022-2496/©2016ElsevierInc.Allrightsreserved.

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 81
informationcontainedinbrainactivitypatterns(Harrison&Tong, modelcanbepositivelyrelatedtoaninformation-basedmetricof
2009; Haynes & Rees, 2005; Kamitani & Tong, 2005; Serences & brainprocessing,thenextstepofdeterminingwhetheraparticular
Boynton,2007),aswellasevidenceofsemantictuningproperties
modelprovidesacompellingfitofthehigh-dimensionalbraindata
(Huth,Nishimoto,Vu,&Gallant,2012;Mitchelletal.,2008).The canbedifficulttodemonstrate.
correlational structure of activity patterns has also been used to Inthisreview,wehighlightseveralrecentstudiesthathavesuc-
characterize object representations in the ventral temporal lobe cessfully combined theoretical models with fMRI data, address-
(Haxby et al., 2001; Kriegeskorte et al., 2008). Correlation-based ing diverse questions spanning lower-level perceptual processes
approacheshavealsoprovenusefulfordelineatingthefunctional tohigher-levelcognitiveprocesses.Acentralthemeacrossthese
connectivityofthehumanbrain(e.g.,Honeyetal.,2009),including studiesisthegoalofidentifyingcompellingrelationshipsbetween
restingstatenetworks(Foxetal.,2005),andmorerecentstudies brain,modelandbehavior(seeFig.1(A)),oftenwiththecognitive
ofbrainconnectivityhavebenefittedfromtheapplicationofgraph
modelservingastheintermediaryformappingbetweenbrainand
| theoretical      | models | (Bullmore | &              | Sporns, | 2009)    | and other | model-   |           |          |       |      |            |          |          |     |
| ---------------- | ------ | --------- | -------------- | ------- | -------- | --------- | -------- | --------- | -------- | ----- | ---- | ---------- | -------- | -------- | --- |
|                  |        |           |                |         |          |           |          | behavior. | However, | as we | will | see, there | are many | possible | op- |
| based approaches |        | (Tavor    | et al., 2016). | Such    | advances | in        | analytic |           |          |       |      |            |          |          |     |
tionsandapproachesforestablishingtheselinks,asamodelfit-
methodscontinuetoexpandthewaysinwhichfMRIcanbeused
tedtobehavioraldatamightbeusedtopredictbrainresponses,
tostudythebrain. orbraindatamightbeincorporatedintoamodeltopredictbehav-
Morerecently,researchonnewmathematicalapproachesfor ior.Moreover,intermediateprocessingstepsmaytakeplacebefore
fMRIhasevolvedbeyondthegoalofsimplydevelopingmorepow- linksareestablished,suchasmethodstoreducethehighdimen-
erful analytic methods, to that of integrating and testing cogni- sionalityofbraindatatolower-dimensionalmeasuresthatcanbe
tivemodels.Thismodel-basedapproachtocognitiveneuroscience
moredirectlyrelatedtomodelpredictions.
| represents | an exciting | development |     | that | goes | beyond | the sim- |     |     |     |     |     |     |     |     |
| ---------- | ----------- | ----------- | --- | ---- | ---- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Webeginthisreviewbydiscussinganapplicationofthenor-
| pler goals | of ‘‘brain | mapping’’, | identifying |     | correlations |     | between |     |     |     |     |     |     |     |     |
| ---------- | ---------- | ---------- | ----------- | --- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
malizationmodeltothevisualperceptionoforientation(Brouwer
individualdifferencesandbrainactivity,orinformation-basedap-
|          |                 |     |          |           |          |     |         | & Heeger, | 2011). | In this | work, | fMRI data | from | early cortical | vi- |
| -------- | --------------- | --- | -------- | --------- | -------- | --- | ------- | --------- | ------ | ------- | ----- | --------- | ---- | -------------- | --- |
| proaches | to characterize |     | cortical | function. | Instead, | the | goal of |           |        |         |       |           |      |                |     |
sualareaswasfirsttransformedintointerpretableconstructsusing
model-basedcognitiveneuroscienceliesindescribingthepercep-
|     |     |     |     |     |     |     |     | a multivariate | modeling |     | approach, | and the | normalization |     | model |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | -------- | --- | --------- | ------- | ------------- | --- | ----- |
tualorcognitiveprocessesthatunderliebehaviorinamathemat- wasthenfittedtotheresultingmeasurements.Wethendescribe
icallyprecisemanner,anddeterminingtheneuralprocessesthat anapplicationofmodelsofvisualattentiontofMRIdata(Pratte,
underliethesecomputations.
|           |         |        |      |        |         |        |          | Ling, Swisher, | &   | Tong, 2013). | Here, | a multivoxel |     | pattern | classi- |
| --------- | ------- | ------ | ---- | ------ | ------- | ------ | -------- | -------------- | --- | ------------ | ----- | ------------ | --- | ------- | ------- |
| Cognitive | process | models | have | a long | history | in the | study of |                |     |              |       |              |     |         |         |
ficationapproachwasusedtotransformmultivariatefMRIdata,
humanperformance.Forexample,modelsbasedonsignaldetec-
|     |     |     |     |     |     |     |     | obtained | from | multiple | levels | of the visual | hierarchy, |     | into an |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---- | -------- | ------ | ------------- | ---------- | --- | ------- |
tiontheoryhaveservedasthefoundationforstudyingperception
interpretablemeasureofinformationrepresentation,andthethe-
(Green&Swets,1966),attention(Lu&Dosher,1998)andmemory
oreticalmodelwasfittedtotheresult.Whereasinbothofthese
| (Kintsch, | 1967). | Early cognitive | research |     | also demonstrated |     | that |            |        |       |            |                |     |         |      |
| --------- | ------ | --------------- | -------- | --- | ----------------- | --- | ---- | ---------- | ------ | ----- | ---------- | -------------- | --- | ------- | ---- |
|           |        |                 |          |     |                   |     |      | studies, a | formal | model | was fitted | to information |     | decoded | from |
stochasticaccumulatormodelscanaccuratelypredictpatternsof
themultivariatefMRIsignal(Fig.1(B)),wenextconsiderastudy
choicereactiontimesacrossnumerousbehavioralparadigms(Rat-
|     |     |     |     |     |     |     |     | in which | the time | course | of the | fMRI signal | on  | individual | trials |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------ | ------ | ----------- | --- | ---------- | ------ |
cliff&Rouder,1998;Stone,1960).Whilesomecognitiveprocess
wasincorporatedwithinatheoreticalmodelofbehavioralmem-
modelshavefocusedonidentifyingandquantifyingafewkeypa-
oryperformance(Fig.1(C)),todeterminewhetherthisneuralsig-
rameterstocapturepatternsofcognitiveperformance,othermod-
nalcanleadtomoreaccuratepredictionsoffreerecallperformance
elsrelyongenerallearningprinciplestotraincomplexnetworks
(Kragel,Morton,&Polyn,2015).AnfMRIstudyofcategorization
| with numerous |         | parameters | to perform |            | a cognitive | task.     | For ex- |            |             |          |     |              |     |           |         |
| ------------- | ------- | ---------- | ---------- | ---------- | ----------- | --------- | ------- | ---------- | ----------- | -------- | --- | ------------ | --- | --------- | ------- |
|               |         |            |            |            |             |           |         | highlights | yet another | approach |     | (Fig. 1(D)), | by  | assessing | the de- |
| ample, neural | network | models     | (e.g.,     | McClelland |             | & Rogers, | 2003)   |            |             |          |     |              |     |           |         |
greetowhichcompetingmodelsofbehavioralcategorizationper-
havebeendevelopedtocharacterizehigh-levelprocessesinclud-
formancecanaccountforobservedpatternsofneuraldata(Mack,
ingspeechperception(McClelland&Elman,1986),categorization
Preston,&Love,2013).Finally,wereviewastudyofcognitivecon-
(Ashby&Maddox,1993;Nosofsky&Palmeri,1997),cognitivecon-
trol(Botvinick,Braver,Barch,Carter,&Cohen,2001),andhuman trolthatdemonstrateshowtheapplicationofatheoreticalmodel
memory(Polyn,Norman,&Kahana,2009). tofMRIdatacanrevealnewinsightsaboutneuralprocessingthat
One might expect that the application of theoretical models wouldhavebeenimpossiblewithoutthemodel(Ide,Shenoy,Yu,
&Li,2013).Here,amodelofbehavioralperformanceinthestop-
| to neuroimaging |     | data would | be  | a naturally | obvious | and | fruitful |     |     |     |     |     |     |     |     |
| --------------- | --- | ---------- | --- | ----------- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
signaltaskwasincorporatedwithinthefMRIanalysis(Fig.1(E)),
| endeavor. | However, | most | cognitive | neuroscientists |     | have | not |     |     |     |     |     |     |     |     |
| --------- | -------- | ---- | --------- | --------------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andtheresultssuggestthatthefunctionoftheanteriorcingulate
rushedtomeetthischallengeuntilrecently.Whyhasthisbeenthe
ismorespecificthanhasbeensuggestedbypreviousstudies.
case?Acentralchallengeliesinestablishingstronglinksbetween
Thiscollectionofstudiesdemonstratesboththefeasibilityand
theparametersofacognitivemodelandparticularbrainresponses
embedded within a cognitive experiment. Cognitive models potentialofmodel-basedcognitiveneuroscience.Theapproaches
typically rely on latent constructs of presumed psychological areremarkablydiverse,bothinhowthedataareusedtoinform
processes that must somehow be translated into a predicted the model, and in the technical solutions employed to establish
patternofbrainresponses.Ifthemodelleadstoclearpredictions compelling relationships between brain, model and cognitive
regarding how the univariate BOLD responses should change performance (see Fig. 1). Of particular interest is that fact that
|                |      |          |             |      |                 |              |         | none of the        | reviewed | works | share | exactly    | the     | same strategy | to      |
| -------------- | ---- | -------- | ----------- | ---- | --------------- | ------------ | ------- | ------------------ | -------- | ----- | ----- | ---------- | ------- | ------------- | ------- |
| over time,     | such | models   | may be      | more | straightforward |              | to test |                    |          |       |       |            |         |               |         |
|                |      |          |             |      |                 |              |         | link a theoretical |          | model | with  | functional | imaging | data.         | Rather, |
| using standard | fMRI | analysis | procedures. |      | Earlier         | applications | of      |                    |          |       |       |            |         |               |         |
theseexamplesunderscorehowindividualstudieshavereliedon
| model-based | fMRI | have | relied on | such | approaches | to  | identify |     |     |     |     |     |     |     |     |
| ----------- | ---- | ---- | --------- | ---- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
the neural correlates of reward prediction error (e.g., O’Doherty, cleverinnovationsthatarecustom-builtforaparticularmodelor
Dayan, Friston, Critchley, & Dolan, 2003; O’Doherty et al., experimentalparadigm.Assuch,weneitherforeseenorprescribea
2004; O’Doherty, Hampton, & Kim, 2007) and response conflict one-size-fits-allapproachtomodel-basedcognitiveneuroscience.
(e.g., Botvinick, Cohen, & Carter, 2004). However, fMRI data Instead, we believe that the diversity in attempts to integrate
is very high-dimensional, such that establishing links between modelingandfunctionalbrainimagingwillforwardtheadvance
cognitive models and the information contained in multivariate oftheoreticalmodelswithamomentumthatwouldnothappenif
brainactivitypatternsisconsiderablymorechallenging.Evenifa thesefieldsremainedisolatedfromoneanother.

82 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
Fig.1. ApproachesToLinkingModelsWithBrainandBehavior.(A)Thegoalofamodel-basedcognitiveneuroscienceistodeveloptheorybyforminglinksbetweenbrain
activity,behavior,andatheoreticalmodelthatdescribesthesedata.(B)IntheapproachtakenbyBrouwerandHeeger(2011)andPratteetal.(2013),amultivariatepattern
analysisisfirstusedtotransformthebrainactivitydataintoalower-dimensionalspace,andthemodelisthenappliedtothesemoreinterpretabledata.(C)Intheapproach
takenbyKrageletal.(2015)thebrainactivitysignalisdirectlyincorporatedwithinthemodel,andstatisticaltestsareusedtodeterminewhetherthisinclusionproducesa
moreaccurateaccountofthebehavioraldatathanamodelwithouttheneuraldata.(D)Macketal.(2013)firstfittedamodeltobehavioraldata,andusedthefittedmodel
todecodepatternsofbrainactivity.Theythencomparedthisdecodingperformanceacrossdifferentmodelstodeterminewhichmodelprovidedthebestcorrespondence
withtheneuraldata.(E)Ideetal.(2013)firstfitedamodeltobehavioraldata,andthenusedaregressionanalysistoidentifyfMRIvoxelsthatco-variedwithvariousmodel
constructsoverthecourseoftheexperiment.
2. Perception:thenormalizationmodel changes size. Neural responses in many brain areas, such as the
lateralgeniculatenucleus(LGN)andstriatecortex(V1),increase
Amajorgoalinthestudyofvisualperceptionistounderstand monotonicallyasafunctionofstimuluscontrast.Theseresponses
howneuralresponsestoastimulusareaffectedbythepresenceof typically follow a sigmoidally shaped response function, with a
other stimuli in the visual field. One of the most widely studied signaturecompressiveeffectathighstimuluscontrasts(Fig.2(A)).
| models | of such | stimulus | interactions, | the | normalization |     | model        |                   |                 |             |
| ------ | ------- | -------- | ------------- | --- | ------------- | --- | ------------ | ----------------- | --------------- | ----------- |
|        |         |          |               |     |               |     | For example, | consider a neuron | in the LGN that | responds to |
(seeCarandini&Heeger,2012forareview)hasbeenrigorously
stimulationwithinitsreceptivefield,inamannerthatdependson
tested using behavioral, neurophysiological, and more recently, ( )
|     |     |     |     |     |     |     | thestimuluscontrast | C .Accordingtothenormalizationmodel, |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | ------------------------------------ | --- | --- |
humanneuroimagingapproaches.Thenormalizationmodelposits
theactivitywithinapoolofnearbycellsthatalsorespondtothis
( )
that the response of a neuron to a stimulus will depend on stimulus C will divisively dampen the response of the target
( ) m
| its sensitivity | to  | the stimulus, | and | further | that this | response | is neuronR | C , |     |     |
| --------------- | --- | ------------- | --- | ------- | --------- | -------- | ---------- | --- | --- | --- |
divisivelyreducedbytheconcurrentactivityofotherneighboring
C
| n e ur o n s . | Th e a ct | i v it y o | f th e s e o t h er | n e u r o ns | i s d r | iv e n l a rg | el y by R ( C )= |     |     |     |
| -------------- | --------- | ---------- | ------------------- | ------------ | ------- | ------------- | ---------------- | --- | --- | --- |
σ +
th e p r e s en ce o f o t h e r st im u l i ,s u c h th a t t h e m o d e l a c co u n ts f or a C m
myriad of behavioral and neural effects resulting from stimulus σ
|               |     |           |          |         |     |               | whereparameter | determinestheshapeofthecontrast-response |     |          |
| ------------- | --- | --------- | -------- | ------- | --- | ------------- | -------------- | ---------------------------------------- | --- | -------- |
| interactions. | The | purported | benefits | of such | a   | normalization |                |                                          |     |          |
|               |     |           |          |         |     |               | function.1     | Thepopulationofneuronsinthedenominator   |     | ( C ) is |
process to the perceptual system include reducing the overall m
referredtoasthenormalizationpool,andthesummedresponses
neuralandmetabolicdemandswithinalocalregion,de-correlating
theresponsesofneuronswithdifferentstimuluspreferences,and
providingformoreefficientneuralcoding.
| Before | considering |     | interactions | between | multiple | stimuli, | a   |     |     |     |
| ------ | ----------- | --- | ------------ | ------- | -------- | -------- | --- | --- | --- | --- |
1 Thesemodelsoftenincludeparameterssuchasanintercept,scaleandnon-
| simpler | case of | normalization | can be | understood |     | by considering |     |     |     |     |
| ------- | ------- | ------------- | ------ | ---------- | --- | -------------- | --- | --- | --- | --- |
linearitytoaccountforthevariousmeasurementsthatareused.Theseparameters
| how the | response | of  | a single neuron | changes |     | as a stimulus |     |     |     |     |
| ------- | -------- | --- | --------------- | ------- | --- | ------------- | --- | --- | --- | --- |
areomittedhereforsimplicity.

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 83
Fig.2. NormalizationModelofPerception.(A)ResponsesofanLGNneurontoastimuluswithinitsreceptivefield,plottedasafunctionofstimuluscontrast(DatafromBonin
etal.,2005,figureadaptedfromCarandini&Heeger,2012).Colorsdenoteresponsesforvariousstimulussizes;linesshowfitsofthenormalizationmodel.(B)Exampleof
cross-orientationsuppressionstimuliusedbyBrouwerandHeeger(2011),wheretargetsofvariouscontrastarepresentedalone(top)orwithasuperimposedmask(bottom).
(C)Orientationtuningcurvesthatserveasbasisfunctionsfortheforwardencodingmodel.Theorientation-tunedresponseofeachvoxelismodeledasaweightedsumof
theseresponsecurves,withweighsestimatedfromthedata.(D)Target-tunedresponsesinV1asafunctionoftargetcontrastpresentedalone(openpoints)orwitha
superimposedmask(filledpoints)fromBrouwerandHeeger(2011).(E)Mask-tunedresponsesinV1asafunctionoftargetcontrastwhenthemaskwasnotpresented(open
points)andwhenitwassuperimposedonthetarget(filledpoints).PanelsDandEadaptedfromBrouwerandHeeger(2011).(Colorfiguresavailableintheonlineversion
ofthisarticle.)
oftheseneuronsacttonormalizetheresponseofthetargetneu- populationofneuronswithresponsesthatdependonthecontrast
ron.Whentheresponseofaneuronisplottedasafunctionofcon- ( C ) andtheorientation( θ )ofastimuluswithinthepopulation’s
trast,increasedactivityintheneuron’snormalizationpoolwillact receptive field, where responses to various orientations are
tosuppressresponsesofthetargetneuronbyanamountthatde- describedbysomefunctionf (θ) .Theresponsesofnearbyneurons
pendsonthestimuluscontrast.Forexample,Fig.2(A)showsthat willalsodependonthecontrastandorientationofthestimulus
the response of an LGN neuron is suppressed as the stimulus is accordingtotheirowntuningcurvesf (θ) .Thesetuningcurves
m
enlargedfrom2°to20°,growingbeyondthatneuron’sreceptive
areincorporatedintothenormalizationmodelbyassumingthat
field(Bonin,Mante,&Carandini,2005).Accordingtothenormal- theresponseofaneurontoaparticularstimulusisweightedby
izationmodel,alargerstimuluswillactivatemorenearbyLGNneu- the neuron’s tuning function and divisively normalized by the
ronsthatmakeupthetargetneuron’snormalizationpool,thereby responseofneighboringneuronstothatsamestimulus,
suppressingtheresponseofthetargetneuron.Whenthestimu- (θ)
f C
lusismadetobesmallerthan2°,theresponseagaindecreasesas R ( C ,θ)= σ +
f
(θ)
C
.
thestimulus nolongeroccupies theentirereceptive field. Over- m m
all, the relationship between stimulus size and the response of Amajorgoalofthenormalizationmodelistoaccountforneural
theneuronisnon-monotonicathighstimuluscontrasts,andthe responses to simultaneously presented stimuli. A prominent
normalizationmodelprovidesanaccurateaccountoftheshapeof exampleiscross-orientationsuppression,inwhichresponsestoa
thecontrast-responsefunction,andthecomplexwaysinwhichits preferred target orientation are suppressed by the presence of
shapechangesacrossmanystimulusmanipulations. a masking grating with an orthogonal orientation. For example,
The normalization model not only accounts for the neural consider the case when a vertical target stimulus of various
interactionsthatmaybeevokedbyasinglestimulus,itprovidesa contrasts is presented alone, or superimposed with a horizontal
parsimoniousaccountofthenon-linearinteractionsthatcantake maskof50%contrast(Fig.2(B)).Thenormalizationmodeliseasily
placewhenmultiplestimuliarepresented,suchassuperimposed amended to account for this situation. Let C and C denote the
1 2
θ θ
gratings of differing orientation (see Fig. 2(B)). Consider a target contrastoftargetandmask,respectively,and and denotetheir
1 2

84 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
orientations. The total response of the target population to both decompose multivariate V1 activity into target-selective and
stimulibecomes: mask-selective responses. This approach therefore provides a
f (θ ) C +f (θ ) C starting point for further investigations of the normalization
R
(
C 1
,
C 2
,θ
1
,θ
2
)=
σ +
f m
(θ
1
1
)
C
1
1,m
+
2
f m
2
(θ
2
)
C 2,m
.
m
&
o
H
d
e
e
e
l
g
,s
e
u
r,
ch
20
a
0
s
9
it
)
s
a
a
n
p
d
pli
t
c
o
at
h
io
ig
n
h
t
e
o
r-
o
l
t
e
h
v
e
e
r
l
f
p
ea
ro
tu
ce
re
ss
d
e
o
s
m
su
ai
c
n
h
s(
a
e
s
.g
a
.
t
,
t
M
en
o
t
r
i
a
o
d
n
i
The model simply states that the total response reflects (e.g.,Herrmann,Heeger,&Carrasco,2012).
the summed response to both target and mask, but that this
responseisdampenedbythesummedresponsesofneuronsinthe
3. Attention:theperceptualtemplatemodel
normalizationpooltobothstimuli.
Brouwer and Heeger (2011) tested the predictions of the
Theeffectofattendingtoaparticularstimulusisoftencharac-
normalizationmodelforcross-orientationsuppressioninhumans
terizedasenhancingtheperceivedqualityoftheattendedstimu-
usingfMRI,followingthestepsshowninFig.1(B).Eachmeasured
lus.Aprominentexampleofthiseffectiswhentheto-be-attended
fMRIvoxelreflectsthepooledresponsefrommanyneurons,and
stimulusisembeddedinexternalnoise,suchastryingtofocuson
in an early visual area such as the primary visual cortex (V1),
aperson’svoiceatacrowdedpartyortryingtoseethrougharain-
mostneuronsexhibitstrongselectivityforlocalorientation.The
spatteredwindshield.Inanefforttodescribethemechanismsthat
orientation-selectiveresponseofavoxelthatencompassesmany
underlieourabilitytoenhancetherepresentationofanattended
such neurons can be described by a combination of the tuning
stimulusinthepresenceofperceptualnoise,LuandDosher(1998)
curves of the underlying neurons. Brouwer and Heeger (2011)
constructed a model based on core concepts of signal detection
developedaforwardmodelingapproach(Brouwer&Heeger,2009;
theory and signal processing, which they termed the Perceptual
Kay et al., 2008) in which the responses of a voxel to different
TemplateModel(PTM).ThePTMmodelassumesthatattendingtoa
orientationsaremodeledasalinearcombinationoforientation-
noisystimulushastheeffectofincreasingthesignal-to-noiseratio
tunedbasisfunctions.Forexample,BrouwerandHeegermodeled
oftherepresentationofthatstimulus.Assuch,attentioncandoone
orientationresponsesusing6evenlyspacedraisedcosineprofiles
oftwothings:increasethestrengthoftheincominginput(amplifi-
tocoverthespaceofallorientations(Fig.2(C)),andtheamplitude
cation),ordecreasetheirrelevantnoise(noisereduction).Whereas
ofresponseforeachofthese6tunedpopulationscouldthenbe
theamplificationmechanismsimplyincreasesresponsestoallof
estimated for each voxel by using linear regression applied to a
theincominginformation(i.e.,bothsignalandnoise),thenoise-
subset of the fMRI data. The fitted model is then inverted and
reductionmechanismselectivelydecreasesresponsestotheexter-
appliedtovoxelresponsesinanotherdataset,providinganoverall
nalnoise.Todoso,themodelassumesthatthereexistsaperceptual
measure of how strongly particular orientations are activated
templateoftherelevantsignal,whichallowsthebraintodiscrim-
acrosstheentirepopulationofvoxelsinV1.Thisanalysisallowed
theresearcherstoseparatelymeasurethestrengthofV1responses inatebetweensignalandnoise.Sinceitsinception,thismodelhas
to various orientations. Consequently, in a cross-orientation beenappliedandtestedbehaviorallyinmanycontexts(seeCar-
scenarioinwhichtwostimuliarepresentedsimultaneously,the rasco,2011forareview).Morerecently,thePTMmodelhasbeen
combined response of voxels in V1 could be decomposed to used with fMRI to determine the neural correlates of these pur-
provideestimatedresponsestoeachstimulusseparately. portedattentionalprocesses.
Intheirexperiment,BrouwerandHeeger(2011)showedver- In the vast majority of fMRI studies, the effect of attention
ticallyorientedtargetstimuliofvariouscontrastseitheralone,or is to increase the mean BOLD signal, for example, within
superimposedwithahorizontallyorientedmaskstimulusof50% retinotopic visual areas corresponding to the attended part of
contrast(Fig.2(B)).Orientationpreferenceswerefirstdetermined space (e.g., Buracas & Boynton, 2007; Kastner, Pinsk, De Weerd,
forvoxelsinV1,andthenthisestimatedforwardmodelwasused Desimone,&Ungerleider,1999).ThePTMmodel,however,makes
toprovideseparatemeasuresofV1responsestotargetandmask theoppositepredictionforaspecificregionofthestimulusspace:
stimuli.Fig.2(D)showstheresultingcontrastresponsefunctions Ifattentionactsasanoise-reductionmechanisminabrainregion,
forV1responsestothetargetorientation,andthefittednormal- thenattendingtoalow-contraststimulusembeddedinhighlevels
izationmodel.Intheabsenceofamask,thetarget-tunedresponse ofexternalnoiseshouldreducethetotalresponsetosignalplus
rises with target contrast in a way that is accounted for well by noise, as responses to the noise are reduced but responses to
themodel.Theadditionofthesuperimposedhorizontalmaskre- the signal are not affected. If the mean BOLD signal in a region
ducestarget-tunedresponses,andthisgrosseffectisthesignature reflects the total activity of cells responding to both signal and
ofcross-orientationsuppression.Specifically,theeffectofthemask noise,thenattentionshouldreducemeanBOLDresponsesforlow-
istoreducetarget-tunedresponsesforlow-contrasttargets,butit contrastsignalspresentedinnoise.Lu,Li,Tjan,Dosher,andChu
hasaweakersuppressiveinfluencewhenthetargetgrating’scon- (2011)foundsupportforthisprediction,showingthatattending
trastexceedsthatofthemaskinggrating,aspredictedbythenor- to low-contrast gratings in high levels of external noise led to
malizationmodel. reducedmeanBOLDresponsesinV1,whereasattentionincreased
Fig. 2(E) shows V1 responses to the mask-tuned orientation, responsestothelow-contrastgratingwhenitwaspresentedalone.
asafunctionoftargetcontrast.Intheabsenceofamask(Target These results illustrate that it is possible to test theoretical
Only condition), these responses are invariant to target contrast models with fMRI by making predictions about mean BOLD,
asexpected.However,thepresenceofthehorizontalmaskleads especially by exploring predicted interactions with parametric
to large horizontally tuned responses for low target contrasts, manipulations of the stimulus. However, the formal PTM model
andtheseresponsessystematicallydecreaseasthetargetcontrast specifieshowamplificationandnoisereductionaffectthesignal-
is increased. This negative-going curve is also predicted by the to-noiseratioofstimulusinformation,ratherthanthetotalneural
normalization model: as the target-tuned cells become more responseasismeasuredwithmeanBOLD.Inparticular,consider
stronglyactivatedbyincreasingtargetcontrast,thesetarget-tuned a stimulus with contrast C that is embedded in visual noise
responseswillacttonormalizeresponsestothemask,leadingto (termed external noise) with contrast N . The goal is to predict
E
greaterinhibitionofthemask-tunedcellsathightargetcontrasts. some measure of the sensitivity to a signal relative to the noise
′
The normalization model provides a simple and accurate ac- (d)foraneuralpopulationtunedtosomepropertyofthestimulus
countofhowfMRIresponsesbehaveinthisrathercomplexstim- (e.g. orientation). Sensitivity varies as a function of the contrast
ulus paradigm. This study relied on a sophisticated approach to of the stimulus and the contrast of the noise. In the absence of

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 85
attention,sensitivityisproportionaltothestimulusstrength,and whether the stimulus was attended to or not. The influence
loweredbyexternalnoise: of attention, however, depended on the external noise level:
Attentionenhancedclassificationperformancewhenstimuliwere
C
′( , )=
d C N E  embedded in high levels of noise, but had no effect for stimuli
N2+N2
presentedinlowormoderatenoiselevels.Thispatternofresults
E I
|     |     |     |     |     |     |     |     | implies | that a pure noise-reduction |     | mechanism |     | operates | in V1: |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------------------------- | --- | --------- | --- | -------- | ------ |
whereN isthelevelofinternalnoiseinherentintheperceptual
I attentionisonlyeffectiveinthepresenceofhighlevelsofexternal
system, and determines the shape of the response which varies noise.Thiscanbecontrastedwiththeresultsfoundinthefusiform
as a function of stimulus contrast (C) or external noise contrast face area (FFA, Fig. 3(C)), an area in the ventral temporal cortex
(N E ).Withinthedenominator,externalandinternalnoisesources thatshowsstrongselectivityforfaces(Kanwisher,McDermott,&
are combined, and the combined effects act to degrade the Chun,1997).UnlikeV1,attentionincreasedthesensitivityofthe
representationofthestimulus. FFAtostimulipresentedinbothlow-andhigh-noiseregimes.The
AccordingtothePTMmodel,therearetwowaysthatattention attentionalenhancementobservedacrossallnoiselevelssuggests
canactontheincomingsignalandnoise.Anamplificationmecha- thatboth noise-reductionandamplificationmechanisms impact
nism(a)willmultiplicativelyenhancetheentireinputsignal,thus the contents of visual representations in these high-level brain
increasingresponsestoboththesignalandtheexternalnoise.Asa
regions.
consequence,amplificationcanimprovetherepresentationofthe
|     |     |     |     |     |     |     |     | The lines | in Fig. 3(B) | and | (C) show | predictions | of  | the fitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------ | --- | -------- | ----------- | --- | ---------- |
stimulusifexternalnoiserangesfromnegligiblelevelstolevelsnot
|     |     |     |     |     |     |     |     | PTM model, | which captures | the | effects | of both | external | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------------- | --- | ------- | ------- | -------- | ----- |
much greater than that of internal noise. In contrast, a selective andattentiononfMRIclassificationperformancequitewell.The
noise-reductionmechanism(r)willactsolelytodecreasethelevel modelwasfittedtodatafromseveralbrainregionsinthevisual
ofexternalnoisewithoutaffectingthestimulus: cortex, providing estimates of noise reduction and amplification
ineach.Fig.3(D)and(E)showstheresultingparameterestimates
| ′( , )= |     | aC  | .   |     |     |     |     |     |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
d C N  ofnoisereductionandamplification,respectively.Whereasnoise
| E   |    | 2  |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
aNE +N2 reduction appears to occur throughout the early visual cortex,
|            | r     | I      |             |     |       |               |     | amplificationisabsentinearlyareasV1andV2,butbecomesmore |                 |        |       |         |      |           |
| ---------- | ----- | ------ | ----------- | --- | ----- | ------------- | --- | ------------------------------------------------------- | --------------- | ------ | ----- | ------- | ---- | --------- |
|            |       |        |             |     |       |               |     | prominent                                               | in higher-level | areas. | These | results | have | important |
| This model | makes | strong | predictions |     | about | how attention |     |                                                         |                 |        |       |         |      |           |
shouldaffectsensitivitytoasignalinapopulationofcells.Anoise- implications for how attention may operate to improve visual
reduction mechanism would significantly improve the quality processing,firstbyrelyingonattentionaltemplatestunedtothe
underconditionsofhighexternalnoisebyreducingresponsesto targettofilteroutexternalnoise,andsubsequentlybyamplifying
thenoise,butintheabsenceofnoise,anoise-reductionmechanism thesefilteredresponsestoenhanceoverallresponses,effectively
wouldhavenothingtodoandwouldleadtononetbenefit.This overcomingtheeffectsofinternalnoise.
Despitetacklingaverydifferentproblem,themethodological
| pattern may   | be contrasted |     | with  | the effects  | of     | a non-selective |     |          |              |            |              |     |          |          |
| ------------- | ------------- | --- | ----- | ------------ | ------ | --------------- | --- | -------- | ------------ | ---------- | ------------ | --- | -------- | -------- |
|               |               |     |       |              |        |                 |     | approach | of Pratte et | al. shares | similarities |     | with the | previous |
| amplification | mechanism,    |     | which | will improve | visual | processing      |     |          |              |            |              |     |          |          |
studybyBrouwerandHeeger(2011).Bothreliedonananalytic
| in low levels  | of external | noise       | and    | comparatively |               | high           | levels of |            |                              |          |       |            |           |          |
| -------------- | ----------- | ----------- | ------ | ------------- | ------------- | -------------- | --------- | ---------- | ---------------------------- | -------- | ----- | ---------- | --------- | -------- |
|                |             |             |        |               |               |                |           | approach   | to isolate stimulus-specific |          |       | components | of        | the fMRI |
| internal       | noise: This | is achieved | by     | boosting      | the           | representation |           |            |                              |          |       |            |           |          |
|                |             |             |        |               |               |                |           | responses, | and then                     | compared | these | observed   | responses | with     |
| of both signal | and         | external    | noise, | such          | that internal | noise          | has       |            |                              |          |       |            |           |          |
thosepredictedbyafittedcognitivemodel.InPratteetal.’sstudy,
lessofanimpact.However,amplificationwillbeineffectivewhen
|          |              |            |      |          |        |         |     | a multivariate | classification | approach |     | was used | to estimate | the |
| -------- | ------------ | ---------- | ---- | -------- | ------ | ------- | --- | -------------- | -------------- | -------- | --- | -------- | ----------- | --- |
| external | noise levels | are higher | than | internal | noise, | as both | the |                |                |          |     |          |             |     |
signalandnoisewillbeincreased,leadingtononetbenefit. amountofstimulusinformationinavisualarea,whichcouldthen
In order to assess the correspondence between this model’s be mapped onto the predictions and specific parameters of the
predictions and fMRI responses, Pratte et al. (2013) adopted PTMmodel.Webelievethatusingmultivariatepatternanalyses
a multivariate pattern classification approach to measure the as a transformational step between the dense spatiotemporal
amount of stimulus information contained at multiple levels of information in the original fMRI data and the predictions of a
formalmodelprovidesapowerfulframeworkforfuturestudies.
thecorticalvisualhierarchyabouttheviewedstimuluscategory.
However,aswewilldiscussbelow,thereareotherpromisingways
| The accuracy | of  | fMRI classification |     | served | as  | a measure | of  |     |     |     |     |     |     |     |
| ------------ | --- | ------------------- | --- | ------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
tolinkfMRIresponseswiththeoreticalmodels.
discriminationsensitivitytolinktothepredictionsofthemodel
(seeFig.1(B)).
Thestimuliwerecomprisedoflinedrawingsoffaces,houses, 4. Memory:thecontextmaintenanceandretrievalmodel
| shoes and | chairs embedded |     | in varying | levels | of  | external | noise |     |     |     |     |     |     |     |
| --------- | --------------- | --- | ---------- | ------ | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
(Fig.3(A)).Participantseitherattendedtothestimulibyperform- Theoretical modeling has played a central role in the study
ingaone-backmatchingtaskontheimages,ortheyperformeda of memory (see Raaijmakers & Shiffrin, 2002 for a review). At
demandingtaskonletterspresentedatfixation,effectivelywith- thesametime,humanneuroimaginghasbeenusedsuccessfully
drawingattentionawayfromtheobjectimages.Tomeasureneural to examine various mnemonic processes including long-term
sensitivitytothestimuliinagroupofvoxels,amultivariatepattern memory(seeHenson,2005;Wagneretal.,1998),freerecall(Polyn,
classifierwastrainedtoidentifywhichstimulustype(face,house, Natu,Cohen,&Norman,2005)andvisualworkingmemory(Ester,
|     |     |     |     |     |     |     |     | Serences, | & Awh, 2009; | Harrison | & Tong, | 2009; | Marois, | 2016; |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------ | -------- | ------- | ----- | ------- | ----- |
shoeorchair)wasbeingviewedinagivenblock,basedonthemea-
suredfMRIactivitypatternsineachcorticalvisualarea.Theability Sprague, Ester, & Serences, 2014; Xu & Chun, 2006). However,
topredicttheviewedstimuluscategorybasedonactivitypatterns bridging these approaches by applying formal memory models
servedasameasureofstimulus-specificsensitivity.Thismeasure to neural data has been largely absent. One reason may be that,
ofclassificationaccuracywasassessedseparatelyforeachexternal unliketherelativelysimplenormalizationandperceptualtemplate
noiselevelandeachattentioncondition. models considered above, memory models are often extremely
Fig.3(B)showsclassificationperformanceforV1asafunction complex.Forexample,neuralnetworkmodelsofmemorycontain
of external noise level, separately for when the stimulus was multiplelayers,manyunitsandnumerousweightedconnections,
attended and when attention was withdrawn from the object such that its performance cannot be expressed in a closed-form
images. As expected, adding external noise to the stimulus equation. As a consequence, their model predictions can only
systematically lowered classification performance, regardless of be approximated via simulation, and their parameters must be

86 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
Fig.3. PerceptualTemplateModelofAttention.(A)ExampleofstimuliusedinPratteetal.(2013),wherehereahouseimageisembeddedinvariouslevelsofexternalnoise.
(B)AccuracyofapatternclassifiertrainedonV1responsestoidentifythecategoryofaviewedobject(face,house,chairorshoe)plottedasafunctionofexternalnoiselevel,
separatelyforwhentheobjectswereattended(filledpoints)orun-attended(openpoints).Linesdenotethefittedperceptualtemplatemodel,andstarsindicatesignificant
attentioneffectsateachnoiselevel.(C)ClassificationaccuracyforactivitypatternsintheFFA.(D)Estimatesofnoisereductioneffectsofattentioninvariousvisualareas
obtainedbyfittingthePTMmodeltoclassificationdatafromvariousregionsofinterest,arrangedlooselyfromearlyvisualareastohigher-levelvisualareas.(E)Estimates
ofamplificationeffectsofattention.
Source:FiguresadaptedfromPratteetal.(2013).
fitted with advanced minimization routines such as differential alistofitemstostudyandarethenaskedtorecallasmanywords
evolution, or after extensive training of the network. With such astheycaninwhateverordertheychoose.Severalcharacteristics
complexity,applyingthesemodelseventosimplebehavioraldata offreerecalldatahavebeenpivotalfortheorydevelopment,such
canbechallenging,suchthatthetaskoflinkingmodelpredictions astheprominenceofprimacyandrecencyeffects.Oneparticularly
toneuraldataisnotstraightforward. interestingfeatureoffreerecallisthestronginfluenceoftemporal
Recentlyhowever,Krageletal.(2015)successfullyestablished context:uponrecallingawordattest,peoplearemorelikelytore-
linksbetweenawell-studiedmodeloffreerecallandfMRImea- callwordsthatwerenearbyinthestudylist(withthesubsequent
suresofbrainactivity,providinginsightsintoneuralmechanisms word more likely to be reported than the preceding word). This
underlying memory processing that would not have been possi- patternofbehavioraldatahasbeeninterpretedtosuggestthatdur-
bleotherwise.Achallengeinunderstandinghumanmemoryisthe ingthestudyofaword,thecharacteristicsofthewordarestored
organizationandstructurethatpeoplenaturallyimposeuponany alongwiththetemporalcontextinwhichthewordwaspresented.
setoflearneditems.Thisstructureisparticularlyevidentinstudies Atretrieval,theactivationofamemorytraceforaparticularword
thatallowforfreerecall,inwhichparticipantsarepresentedwith willalsoactivatethecontextwithwhichthatitemwasstored,and

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 87
thisheightenedactivityhelpsbringtomindotherstudieditems The ability to tease apart these cognitive processes relied
thatoccurrednearbyintime. on incorporating fMRI activity within a theoretical model of
This notion of temporal reinstatement has been formalized contextual reinstatement. The result of doing so provides new
in the Context Maintenance and Retrieval Model (CMR, Polyn evidence for the role of the medial temporal lobe in episodic
et al., 2009), a member of a large class of temporal context memoryandfreerecall,bylinkingBOLDactivityintheseregionsto
models(Howard&Kahana,2002).AccordingtotheCMRmodel, formalizedcomputationalmechanismsoftemporalcontext.These
a studied item can be represented by a feature vector, as is resultsalsoprovidesupportfortheCMRmodel:Ifthemodelwere
standard in many models of memory. Additionally, a separate poorlyalignedwithcognitive/neuralprocesses,thenitisunlikely
vectorrepresentsthecontextatthetimeofstudyforagivenitem. that its parameters would map onto brain responses in such a
Thecontextsurroundingawordatstudyincludesthepreviously structuredandcoherentmanner.Othertemporalcontextmodels
studiedwords,andwillbemoregreatlyinfluencedbymorerecent may fair better or worse at linking the BOLD signal to behavior,
words.Consequently,thiscontextvectorchangesgraduallyover andthemodelingapproachdevelopedbyKrageletal.providesa
the course of the study epoch, and serves to link the memory powerfulframeworkforaddressingsuchmodel-basedquestions.
representationsoftemporallyproximalwords.Attest,recallingan TheapproachtakenbyKrageletal.toestablishlinksbetween
itemwillre-activatethecontextstateassociatedwiththatitem, atheoreticalmodelandneuraldataisquitedifferentthanthose
thusincreasingtheprobabilityofrecallingwordsthatwerestudied discussedpreviously,whereamodelwasfittedtoamplitudesof
nearby. fMRI population responses or to the accuracy of fMRI decoding.
Kragel et al. (2015) posited that changes in brain activity Instead,Krageletal.incorporatedfMRIsignalstomodulatelatent
duringtherecallperiodmayreflectthesedynamics,drivenbytwo processeswithinamodelofbehavioraldata,andidentifiedvoxels
differentprocesses:(1)Moreorlessbrainactivityinanareamight for which doing so led to better behavioral predictions. This
correspondtohowwellawordisbeingrecalled,termedretrieval approachservesasapromisingframeworkforfuturestudies,and
success,and(2)Activitymayberelatednottotheitemstrength, providesacompellingdemonstrationthattherearemanywaysto
butrathertothedegreewithwhichthecontextsurroundingthe accomplishamodel-basedapproachtocognitiveneuroscience.
itemisbeingactivated,termedtemporalreinstatement.
There is nothing inherent in the experimental design of a 5. Categorization:exemplarvs.prototypemodels
freerecall taskthat can beused todecompose theneural signal
intothesetwoseparateprocessesoftemporalreinstatementand Inthestudiesconsideredsofar,aspecificmodelwasappliedto
general memory performance. Instead, Kragel et al. devised an somecombinationofbehavioralandneuraldata,andtheresulting
approachoffirstfittingastandardCMRmodeltothebehavioral fits and parameter estimates were used to make inferences
data (i.e. the recall events) collected during fMRI scanning, in about the underlying mechanisms. However, formal theoretical
which parameters were found by maximizing the likelihood of modelingprovidesforamorepowerfulapproachoffittingmultiple
the behavioral data given the model. They then fitted enhanced models to the data, such that the merits of different cognitive
versions of the model in which the fMRI signal from a given theories can be compared quantitatively. Admittedly, the task
voxel was allowed to modulate a specific parameter of the CMR of developing multiple models to link to neural data presents
model over the time course of the recall period (see Fig. 1(C)). a greater challenge, especially to ensure judicious and equitable
A voxel’s average response to each recall event could either implementationsofeachmodel.However,arecentstudybyMack
modulatetheoverallsuccessrateofmemoryperformanceduring etal.(2013)demonstrateshowdoingsocanleadtopowerfulways
the course of recall, or it could modify the strength of temporal totestcognitivetheories.
reinstatement during recall. If the resulting model, informed by People are highly adept at learning categorical boundaries
this modulatory source of fMRI signal, was more likely to have in multidimensional spaces, and can accurately categorize new
producedaparticipant’sparticularbehavioraldatathanthemodel instances according to these learned categories. There are two
without fMRI modulation included, then the voxel was deemed prominentclassesofmodelsthatdescribehowpeoplemightdo
to be related to the particular process (accounting for model this, namely prototype and exemplar-based models (see Ashby
complexity using AIC). By allowing the response of each voxel & Maddox, 2005 for a review). Prototype theories posit that as
to affect one, both or neither model parameter, the researchers we encounter multiple instances from different categories, we
couldidentifyvoxelsthatwereinvolvedinmemoryperformance, constructaprototypeforeachcategory.Forexample,theprototype
temporalreinstatement,orboth. may be constructed as the average of all stimuli experienced in
The results of this analysis provided novel evidence to that category. Upon seeing a new exemplar of a category, the
implicatedistinctregionsofthemedialtemporallobeintemporal values of its features are then compared with the prototypes
reinstatement (Fig. 4(A)) and retrieval success (Fig. 4(B)). A for each category. Greater similarity (or proximity) to one of
spatial gradient can be observed along the posterior–anterior the prototypes then provides the basis for categorization of the
axis, whereby more anterior voxels are involved in retrieval novel exemplar. In contrast, exemplar-based theories posit that
success and more posterior voxels co-vary with the level of peoplestoreparticularinstancesfromeachcategoryastheyare
temporal reinstatement. The authors used a statistical model experienced.Uponseeinganewinstancethatistobecategorized,
comparison approach to assess the merit of various models, to thevaluesofitsfeaturesarecomparedwithallpreviousexemplars
determine which regions of interest were related to retrieval stored in memory for each category, and the current instance is
success (RS), temporal reinstatement (TR), or both processes. categorized based on its similarity with all exemplars stored in
Themodelcomparisonresults furthersupportedthenotionof a memory.
posterior–anterior gradient both in the hippocampus (Fig. 4(C)) Mack et al. created stimuli that took on one of two values of
andthemedialtemporalcortex(Fig.4(D)).Takentogether,these color,shape,size,andspatiallocation.Thesetofpossiblestimuli
results suggest that when posterior regions showed increased weredividedintotwocategories,suchthattheaveragevaluesof
neural activity, there was an increased likelihood that the next each stimulus dimension varied across categories, and provided
recalled item would be one that was studied nearby in time. In reasonableprototypesofeachcategory.Participantsweretrained
contrast,activityinanteriorregionsdidnotrelatetothetemporal tocategorizeasubsetofthestimuli,andwerethentestedwithnew
organizationofrecalleditems;instead,loweractivityindicateda instances from each category while undergoing fMRI scanning.
failureinretrievalsuccess. According to prototype theories, the degree to which a new

88 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
Fig.4. ContextMaintenanceandRetrievalModelofMemory.(A)Voxelsthatleadtoabetter-fittingbehavioralmodelwhentheirsignalsmodulatethetemporal
reinstatementparameterofthemodel.(B)Voxelsthatproduceabetterbehavioralmodelwhentheymodulatetheretrievalsuccessparameter.(C)ResultsofanROIanalysis
inwhichtheROIsareorderedfromtheirposterior-to-anteriorpositionwithintheHippocampus.Barsdenotetherelativeevidence(weightedAIC)foreachofthethreefitted
models:(1)amodelinwhichtheROIsignalmodulatesthetemporalreinstatementparameters(TR,red),(2)theROIsignalmodulatestheretrievalsuccessparameters(RS,
blue),or(3)theROIsignalmodulatesbothparameters(Joint,orange).(D)SameROIanalysisasin(C)butforROIswithinthemedialtemporallobe.(Colorfiguresavailable
intheonlineversionofthisarticle.)
Source:FiguresadaptedfromKrageletal.(2015).
instancebelongstoeachcategoryisbasedonhowsimilarthenew Theexemplar-basedmodelfollowsasimilarstructure,however
instanceistotheaverage,orprototype,storedinmemoryforeach similarity is based on the summed distance between a new
category.Ifx km denotestheseaveragesforthekthcategoryandmth instanceandallpreviouslystoreditemsinacategory,ratherthan
featuredimension,andthefeaturevaluesofanewitemattestare distancetotheaverageitem.Ifx jm arethemthfeaturevaluesfor
y forthemthfeaturedimension,thenthesimilaritybetweenthe thejthpreviouslystudieditem,thenthetotalsimilaritybetweena
m
itemandthekthcategoryprototypeis newitemy andtheKthcategorycanbespecifiedby
m
|                                                     |            |                       |              |                 |              |                   |             |
| ---------------------------------------------------- | ---------- | ---------------------- | ------------ | ---------------- | ------------ | ------------------ | ----------- |
|                                                      | 4         |                        |              |                  | 4          |                    |             |
| =exp −c                                              | w |y −x    | |                      |              |                  | w            |                    |             |
| s k                                                  | m m        | km                     |              | s =exp −c        | |y −x        | |                  |             |
|                                                      |            |                        |              | k                | m m          | jm                 |             |
|                                                      | m=1        |                        |              |                  | j∈K m=1      |                    |             |
| where parameter                                      | c controls | how quickly similarity | decreases as |                  |              |                    |             |
|                                                      |            |                        |              | where parameters | and w govern | overall similarity | and feature |
| thedistancebetweenaprototypeandanewitemincreases,and |            |                        |              |                  | m            |                    |             |
weighting,respectively.
| parameters w | allowforparticulardimensionstobedifferentially |     |     |     |     |     |     |
| ------------ | ---------------------------------------------- | --- | --- | --- | --- | --- | --- |
m
weighted.Forexample,ifmoreattentionwerepaidtoonefeature Foreachitemtobecategorizedattest,theseequationsdescribe
dimension over the others, leading to greater weighting of that themodelpredictionsregardingthesimilarity(orrepresentational
feature, then that feature would have a greater influence on match) between the representation of that item and that of
the similarity evaluation ( s ) between an item ( y ) and the each category. These similarity values can be used to compute
|     |     | k   | m   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
prototypes ( x ) for each category. The exponential function a predicted probability that an item is placed within each
km
leadstoasimilarityvalueof1.0whentheinstancematchesthe category, and these predictions can be compared with observed
prototype exactly, and the exponential decay of this similarity categorizationperformancetoevaluatethepredictivepowerofthe
metric provides for a generally good characterization of human models. In their experiment, Mack et al. (2013) found that both
similarityjudgments(Shepard,1987). models fitted the participants’ behavioral performance equally

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 89
Fig.5. ExemplarandPrototypeModelsofCategorization.(A)Categorizationmodelspredictacertainamountofrepresentationalmatchbetweenaninternalrepresentation
andaparticularitemfromalearnedcategory.Herethematchfortheexemplar(green)andprototype(blue)modelsareshownforvariousitemsusedbyMacketal.(2013).
(B)Thelevelofagreement(measuredasmutualinformation)betweentherepresentationalmatchaspredictedbyeachmodel,andthelevelofrepresentationalmatchas
measuredinthefMRIsignal.Valuesofzerowouldindicatenocorrespondencebetweenamodel’spredictionsandthedata;largervaluesindicateahighercorrespondence.
(C)Searchlightanalysistoidentifyareaswherethedifferencesinbrainactivitytothedifferentitemsattestwerewellpredictedbytheexemplarmodel.(Colorfigures
availableintheonlineversionofthisarticle.)
Source:FiguresadaptedfromMacketal.(2013).
well,asisoftenthecase(Rouder&Ratcliff,2004).Althoughthe datacouldbeusedtopredictmatchvaluesasdefinedbythetwo
prototypeandexemplarmodelspredictdifferentrepresentational models provides a way to determine which model more closely
match values for each item (Fig. 5(A)), the coarseness of the matchestheneuralrepresentation(seeFig.1(D)).
behavioral measure (category A or category B) appeared to lack The correspondence between model-based representational
thesensitivitytodiscriminatebetweenthemoresubtledifferences matchvaluesandthosepredictedfrombrainactivitywascaptured
betweenthemodelsintheirrepresentationalmatchpredictions. by a measure of mutual information between predicted and
Mack et al. (2013) evaluated whether neural activity as observedvalues.Thismutualinformationmetricwassignificantly
measuredwithfMRImultivariatepatternanalysismightprovidea greater than zero for both models (Fig. 5(B)), suggesting that
moresensitivemeasureofvariationsinrepresentationalmatchin brainactivity(fromvoxelschosenacrosstheentirebrain)conveys
thiscategorizationtask.Todoso,theyfirstfittheprototypemodel someinformationaboutthesimilaritybetweenapresenteditem
toaparticipant’sbehavioraldata,andusedthemodeltolabeleach and the previously learned category representations as defined
testtrialbyitspredictedrepresentationalmatchvalue(Fig.5(A)). byeithermodel.Criticallyhowever,thismeasurewashigherfor
Inacross-validationprocedure,theyappliedaregressionanalysis theexemplarmodelthanfortheprototypemodel,implyingthat
to a subset of the data, mapping the weighted sum of voxel the exemplar model provides a better description of category
activityontothemodel-definedrepresentationalmatchvaluefor representations in the human brain. A follow up searchlight
eachtrial(using1000voxelschosenfromtheentirebrainusing analysis was conducted to identify brain areas that showed
a localizer analysis). This fitted linear model was then used to activationdifferencestodifferentitemsthatwereinlinewithitem
predict the representational match values of trials not included differencespredictedbytheexemplarmodel(Fig.5(C)).Theresults
inthetrainingdataset,basedonthecorrespondingvoxelactivity indicated that regions of the lateral occipital complex, inferior
patterns.Thisprocedurewasrepeatedsuchthatrepresentational parietal lobe, as well as the right inferior frontal lobe, exhibited
matchvalueswerepredictedforeverytrial,basedontheneural responsesconsistentwiththepredictionsoftheexemplar-based
signal. The accuracy of these predictions served as an index of model.
how closely the neural activity patterns matched the prototype These results underscore the potential power of combining
model’sdefinitionofrepresentationalmatch.Predictionaccuracy theoreticalmodelswithneuroimagingtechniques.Althoughthese
abovechancewouldindicatethattheneuralsignalcarriescategory models make unique predictions about behavioral data, the
information as defined by the prototype model. Critically, the resolution of such data is not always sufficient to provide
sameprocedurewasconductedfortheexemplarmodel,providing convincing evidence in favor of one model or another. By using
predictionaccuracyforrepresentationalmatchvaluesasdefined multivariate pattern analysis to construct an appropriate link
bydistancefromtheexemplars.Comparinghowwelltheneural between the neural data and model parameters, the cortical

90 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
activitypatternsobservedinthisstudyprovidedameasurement
that could be used to adjudicate between the categorization
models.Theabilitytomakeacoherentlinkbetweentheoriesand
brainimagingdataunderliesthesuccessofthisandothermodel-
based neuroscience approaches. Mack et al. (2013) demonstrate
thatdoingsocanprovideinsightsintoprocessingthatwouldnot
haveotherwisebeenpossible.
6. Errormonitoringinthestopsignaltask
Avarietyofcognitivefunctionshavebeenascribedtothedorsal
anteriorcingulatecortex(dACC),basedonstudiesconductedusing
fMRI,EEG,andothermethods.Prominenttheorieshaveproposed
rolesforthedACCinerrordetection,conflictmonitoring,erroran-
ticipation,volatilitymonitoring,action–outcomelearning,reward,
andthecodingoferrorsinpredictingfutureoutcomes,tonamea
few(seee.g.Botvinicketal.,2004;Hayden,Heilbronner,Pearson,
&Platt,2011;Rushworth,Walton,Kennerley,&Bannerman,2004).
TeasingapartwhichoftheserolesthedACCisprimarilyresponsi-
bleforduringataskisextremelychallenging,asmosttasksthatare
designedtotapintooneoftheseprocessesmayalsoevokesome Fig.6. BayesianModelofStopSignalTask.ContrastingtheBOLDsignaltoidentify
combinationoftheothers.Forexample,tasksdesignedtoeliciter- regionsofgreateractivityforstopthanforgotrials(red/orange)revealsextensive
activationinbothcorticalandsub-corticalregions.Themorenuancedmodel-based
rors often do so by incorporating rare events within the experi-
analysis,however,revealsthatonlyasmallregion(magentacolor)withinthedorsal
mentaldesign,suchthateventsthatproducebehavioralerrorsare anteriorcingulatecortex(indexedbycrosshairs,labeleddACC)isrelatedtothe
alsorare.Consequently,brainresponsestotheserareeventsmight processofcomparingexpectedoutcomeswithactualoutcomes,termedsurprise.
reflectthefactthattheyareunexpected,oralternatively,because (Colorfiguresavailableintheonlineversionofthisarticle.)
Source:FigureadaptedfromIdeetal.(2013).
they tend to elicit error responses. Fortunately, formal cognitive
modelshavebeendevisedthatcanprovideseparatemeasuresof
dynamic latent process is measured by parameter P . Another
thesevariousprocesses,basedonthestructureofthetaskandthe stop
parameter,P ,reflectsthelikelihoodthataparticipantwillmake
resultingbehavioraldata.Thekeytodetermininghowthesepro- error
anerrorontheupcomingtrial,dependentupontherecentstimulus
cessesmapontoparticularbrainregions,suchasthedACC,isto
historyandtheparticipant’spastbehavior.Byspecifyinghowthese
somehowleveragethemodelsinordertoisolatetheneuralactivity
andseveralotherlatentprocessesarelinkedtothebehavioraldata,
associatedwithspecificcomponentsoftheseoverlappingmental
themodelisabletopredicttheengagementoftheseprocessesfor
processes.
eachtrialacrossanexperiment.
Ide et al. (2013) set out to isolate these often co-occurring
Ide et al. (2013) fit this model to behavioral data collected
mentalprocesses,suchaserrordetectionandconflictmonitoring,
whileparticipantsunderwentfMRIscanning.Theresultingfitted
in order to determine what brain areas are involved in which
model provided trial-by-trial predictions regarding the levels of
particularfunctions.Theyutilizedaclassiccognitivecontroltask,
knownasthestop-signaltask(Logan,Cowan,&Davis,1984;see multiple latent constructs, such as P stop and P error . These values
Verbruggen&Logan,2008),inwhichparticipantsmakeabutton werethenusedasindependentvariablesinamultipleregression
pressinresponsetoa‘‘go’’signal,suchasagreendot.Onaminority model, in which the BOLD signal from individual voxels served
of trials, however, a ‘‘stop’’ signal (e.g. a red dot) appears some as the dependent variable. The results of the regression analysis
time following the go signal, instructing participants to refrain could then be used to identify voxels that co-varied with each
frommakingaresponsetotheprecedinggostimulus.Replicating particularlatentprocess,suchastheparticipant’sexpectationsof
previous results, Ide et al. (2013) found that the dACC as well astoptrial(seeFig.1(E)).Onecriticalanalysisinvolvedcomparing
as many surrounding regions showed a stronger BOLD response theparticipant’spresumedexpectationsaboutatrial,asspecified
on stop trials than on go trials (yellow/orange colors in Fig. 6). bythemodel,withtheactualoutcomeofthetrial.Inparticular,
However,thisheightenedactivitymightreflectanycombinationof thedifferencebetweenthelatentparameterP stop andtheactual
manyunderlyingcognitiveprocesses.Forexample,higheractivity stimuluscondition(stopvs.go)oneachtrialprovidedameasure
onstoptrialsmightreflecttheprocessingofconflictbetweenthe ofthedegreeofmismatchbetweenparticipant’sexpectationsand
prepotent‘‘go’’responseandtherequiredstoppingbehavior,orit theactualstimulusoutcome,termedsurprise.Themagentacolor
might reflect the experience of surprise associated with the rare in Fig. 6 shows voxels that were modulated specifically by this
eventofastoptrial. measure of surprise, and the results of this whole-brain analysis
TheauthorsappliedaBayesianrationaldecision-makingmodel revealed that voxels sensitive to surprise are located squarely
toaccountforbehavioralperformanceinthestop-signaltask,with withinthedACC(markedbycrosshairs).Thisanalysissuggested
dynamicupdatingbasedontrial-by-triallearning(Shenoy,Angela, thatthedACC’ssensitivitytosurpriseonstoptrialsisindependent
&Rao,2010;Shenoy&Yu,2011).Thedegreetowhichaparticipant of other cognitive processes associated with stop trials, such as
expectsastopsignaltooccuronagiventrialcanbespecifiedby cognitive control and error monitoring. In fact, the researchers
a latent variable ( P ) that varies from trial to trial depending performed several additional analyses to determine what other
stop
on the stimulus history and the participant’s past behavior. For factorsmayco-varywithdACCactivity,butfoundnone.
example, following a long sequence of go trials, people tend to The results of this fMRI modeling work suggest that the
makeprogressivelyfasterresponses.Thisbehavioristhoughtto dACC is involved in comparing our expectations about the
reflect a lowered expectation of an impending stop trial after environment with actual outcomes, and that accounting for this
eachsuccessivegotrial,suchasbythebuildupofanautomatic, cognitive process may preclude the need to invoke additional
prepotentresponse.Inthemodelthisbehaviorisinstantiatedasa cognitiveprocessessuchaserrormonitoringorresponseconflict.
relaxingofthedecisioncriteriontomakea‘‘go’’response,andthis Because these processes are so commingled in typical tasks,

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 91
contrasts between conditions in previous studies likely reflect a severalconsiderations.First,thestimulus,experimentalparadigm,
conglomeration of cognitive processes. However, Ide et al.’s use andtaskmayrequiresomemodificationstobewell-suitedformul-
ofatheoreticalmodelinconjunctionwithfMRIprovidedamore tivariatepatternclassification(seeTong&Pratte,2012fordetails).
nuancedandpowerfulapproachforascribingparticularfunctions Inaddition,specialcaremustbetakentoensurethattheresults
totheanteriorcingulate. ofsuchacomplexintermediaryanalysisservetomeasurethein-
tended construct. In the studies considered here that leveraged
7. Discussion multivariatepatternanalysis(Pratteetal.,2013),multivariatere-
gression(Macketal.,2013)oramultivariateforward-modelingap-
Humancognitiveneuroscienceandtheoreticalmodelingofcog- proach(Brouwer&Heeger,2011),suchconsiderationswerebuilt
nitive processes have led to remarkable advances in our under- intothedesignandanalysisoftheexperiment.However,demon-
standingofmentalfunction,butlargelyasisolatedendeavors.The stratingthatthesecomplexanalysismethodsaremeasuringthe
integrationoftheseapproachesisanaturalstepforward,andthe neuralinformationthattheyareintendedtoextractwillbecrit-
recentproliferationofreviewarticles(Forstmann,Wagenmakers, ical for future studies that utilize such techniques. Nonetheless,
Eichele, Brown, & Serences, 2011), books (Forstmann & Wagen- we believe that multivariate approaches such as pattern classi-
makers,2015),andspecialjournalissuesspeakstothegrowingin- fication analyses (Kamitani & Tong, 2005; Norman et al., 2006;
terestinmodel-basedcognitiveneuroscience.Establishingalink Tong&Pratte,2012)andforwardmodelingapproaches(Brouwer
betweenamodel’sconstructsandnon-invasivemeasuresofhu- &Heeger,2009;Kayetal.,2008)willprovideforpowerfulwaysto
manbrainactivitypresentsmanychallenges.Thestudieswehave linkhigh-dimensionalneuroimagingdatawithmodelparameters
reviewedherehighlightthediversityofrecentapproachestomeet thatisolateimportantcognitiveconstructs.
thesechallenges,andwetaketheirsuccessasevidencethatthere Anothercommonalityamongrecentmodel-basedneuroscience
aremanypromisingparadigmsforbuildingamodel-basedcogni- work is the use of temporally dynamic models. For example, in
tiveneuroscience. the work of Kragel et al. (2015), the Context Maintenance and
The studies reviewed here differ not only in the particular Retrieval model specifies latent constructs such as the strength
methodsused,butdifferinmorefundamentalwayswithrespect of temporal reinstatement over the course of free recall. By
to how the models and neural data were combined (see Fig. 1). incorporatingtheamplitudeofthefMRIsignaloneachtrialinto
For example, some studies fitted a theoretical model directly to specificparametersofthemodel,theycoulddeterminewhether
theneuraldata,andusedtheresultingparameterestimatesand the neurally informed model provided a better prediction of
modelfitstolearnabouttheunderlyingpsychologicalandneural freerecallperformance.Modelssuchasthis,inwhichprocesses
processes (Brouwer & Heeger, 2011; Pratte et al., 2013). Other are proposed to vary over the course of an experiment, provide
studiesfirstfittedamodeltothebehavioraldataandthenusedthe excellent candidates for a model-based neuroscience. For such
resultingfittedmodeltomakeinferencesabouttheneuralsignals models,thebehavioraldataaloneisoftennotsufficienttoidentify
(Ideetal.,2013).Theoppositeapproachhasalsoproveneffective, howprocessesmightvaryovertime.However,thecorresponding
inwhichfMRIsignalsfromavoxelorbrainregionofinterestare neuralmeasurementsprovidepotentiallyinformativeinformation
incorporatedwithinamodelinordertomakebetterpredictions aboutthetemporaldynamicsofsuchlatentprocesses.Although
of behavioral performance (Kragel et al., 2015). Finally, Mack some dynamic cognitive phenomena occur on slow time scales
et al. (2013) fitted multiple models to the behavioral data, and that can be measured with fMRI (e.g., Davis, Love, & Preston,
determined which model produced the highest correspondence 2012; O’Doherty et al., 2003; Serences & Saproo, 2009), such as
with patterns of brain activity. The approaches taken in these variationsinperformanceacrosstrials(e.g.,Ideetal.,2013),the
studiesisnotanexhaustivelist,butthediversityclearlyshowsthat sluggish nature of the hemodynamic BOLD response may limit
therearemanypossiblepathsforintegratingtheoreticalmodeling the ability to study fast cognitive processes that typically occur
and cognitive neuroscience. Going forward, the development of overatimeframeofafewhundredmilliseconds.However,other
newwaystolinkmodelsandneuraldatahasgreatpotentialfor methodsthatmeasureneuralsignalsinrealtime,suchasEEGand
advancing our theoretical and empirical understanding of brain MEG, provide a promising avenue to investigate faster dynamic
function.Althoughwearehesitanttomakestrongprescriptions processesinthesamemanner.
for how one ought to approach model-based neuroscience, it Formal models have provided many insights into cognitive
may be helpful to consider the similarities of an approach with processes,butprogresscansometimesstallfollowingthedevel-
previously successful efforts. Although no two studies reviewed opmentofmoreadvancedandnuancedmodels,whenthebehav-
here completely converged in their methods, there are some ioraldatafailstoclearlydistinguishbetweencompetingmodels.
commonalitiesamongthemthatwehighlightbelowaspromising Thecomparisonofprototypeandexemplarmodelsofcategoriza-
avenuesforfuturework. tionisastrikingexample,asthesemodelsaimtoexplainthesame
Oneapproachforestablishinglinksbetweenhigh-dimensional phenomena,anddifferonlysubtlyinthenatureoftheirunderlying
fMRIdataandparametersofinterestinacognitivemodelistocon- latentrepresentations.Thisfeaturehasmadethemodelsparticu-
structalower-dimensionalmeasureoftheinformationcontained larlydifficulttodifferentiateusingbehavioralmeasuresalone,as
in fMRI activity patterns. This approach was taken in the stud- thelatentconstructsproducesimilarbehavior.However,byusing
iesofthenormalizationmodelofperception(Brouwer&Heeger, the fMRI signal to tap into the underlying latent representation,
2011)andtheperceptualtemplatemodelofattention(Pratteetal., Macketal.(2013)wereabletocomparethemodelswithagranu-
2013). Both studies utilized multivariate pattern analyses as an laritynotpossiblebystudyingbehavioralone.Wedoubtthatthis
intermediarysteptolinkthemultivariateBOLDresponseswiththe isaspecialcase,butrather,wesuspectthatneuralmeasurements
theoretical model. For example, Pratte et al. (2013) applied pat- mightbeinformativetothestudyofavarietyofproposedlatent
ternclassificationtodeterminehowwelltheactivitypatternsin constructsthataredifficulttomeasureaccuratelywithbehavior
individual visual areas could predict which of four possible ob- alone.Thechallengewillbetodesignparadigmsandanalyticap-
ject categories the participant was viewing on individual stimu- proachesthatlinkneuralsignalstotheproposedlatentprocessin
lusblocks.Theresultingpredictionaccuracyprovidedameasureof atestablemanner.Thestudiesreviewedhereinprovideevidence
howrobustlyanobjectwasrepresentedwithinaparticularvisual thatdoingsoishighlyfeasible.
area,whichispreciselythetargetedconstructforthetheoretical Itisoftensaidthatnomodeliscorrect,butthatsomemodels
model.Adoptingamultivariateapproachinthiswaynecessitates arebetterthanothers,andamongthebetterones,somemaybe

92 M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93
considered adequate with respect to their predictive power and Brouwer,G.J.,&Heeger,D.J.(2011).Cross-orientationsuppressioninhumanvisual
theplausibilityoftheirassumptions.Eachofthestudiesreviewed cortex.JournalofNeurophysiology,106(5),2108–2119.
Buckner,R.L.,Bandettini,P.A.,O’Craven,K.M.,Savoy,R.L.,Petersen,S.E.,Raichle,
here included various checks to ensure that the utilized model
M.E.,etal.(1996).Detectionofcorticalactivationduringaveragedsingletrials
wasatleastadequate,suchthatcombiningthemodelwithneural ofacognitivetaskusingfunctionalmagneticresonanceimaging.Proceedings
data allowed for clearly interpretable inferences to be drawn. of the National Academy of Sciences of the United States of America, 93(25),
14878–14883.
These inferences would not have been possible with a poorly
Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical
specified model, and the often-demonstrated match between analysis of structural and functional systems. Nature Reviews Neuroscience,
predictedandobservedneuraldataspeakstothepotentialpower 10(3),186–198.
Buracas,G.T.,&Boynton,G.M.(2007).Theeffectofspatialattentiononcontrast
of these models. Looking forward, we believe that one of the
response functions in human visual cortex. Journal of Neuroscience, 27(1),
mostpromisingavenuesforamodel-basedneurosciencewilllie 93–97.
inusingcognitiveneurosciencedatainordertotest,constrain,and Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural
computation.NatureReviewsNeuroscience,13(1),51–62.
comparedifferentmodels,withtheeventualgoalofdevelopingthe
Carrasco,M.(2011).Visualattention:thepast25years.VisionResearch,51(13),
nextgenerationofbettertheoreticalmodels. 1484–1525.
Ourreviewwasselective,aimingtodemonstratethebreadthof Davis,T.,Love,B.C.,&Preston,A.R.(2012).Learningtheexceptiontotherule:
model-basedFMRIrevealsspecializedrepresentationsforsurprisingcategory
topicsforwhichamodel-basedcognitiveneurosciencehasproven members.CerebralCortex,22(2),260–273.
successful, and the diversity of approaches taken. Other notable Duyn,J.H.(2012).Thefutureofultra-highfieldMRIandfMRIforstudyofthehuman
achievements in combining theoretical models with functional brain.Neuroimage,62(2),1241–1248.
Engel,S.A.(2012).Thedevelopmentanduseofphase-encodedfunctionalMRI
imaginghavebeenmadeinrecentyears,withrespecttothestudy
designs.Neuroimage,62(2),1195–1200.
ofinformationaccumulation(Turner,vanMaanen,&Forstmann, Engel,S.A.,Rumelhart,D.E.,Wandell,B.A.,Lee,A.T.,Glover,G.H.,Chichilnisky,E.
2015), reinforcement learning (Badre & Frank, 2012; O’Doherty J.,etal.(1994).fMRIofhumanvisualcortex.Nature,369(6481),525.
Ester,E.F.,Serences,J.T.,&Awh,E.(2009).Spatiallyglobalrepresentationsin
et al., 2007), and with complex cognitive architectures such as
humanprimaryvisualcortexduringworkingmemorymaintenance.Journalof
ACT-R (Borst & Anderson, 2014). In this review, we chose to Neuroscience,29(48),15258–15265.
focus on fMRI approaches to neuroscience, and on theoretical Forstmann, B. U., & Wagenmakers, E. J. (2015). An Introduction to Model-Based
modelsofperceptionandcognition,butitisworthnotingthatthe CognitiveNeuroscience.NewYork:Springer.
Forstmann,B.U.,Wagenmakers,E.J.,Eichele,T.,Brown,S.,&Serences,J.T.(2011).
moregeneralideasunderlyingmodel-basedneurosciencehavethe Reciprocal relations between cognitive neuroscience and formal cognitive
potentialtoimpactallareasofneuroscienceandpsychology. models:oppositesattract?TrendsinCognitiveSciences,15(6),272–279.
Weanticipatethatfutureadvancesintheoreticalunderstand- Fox,M.D.,Snyder,A.Z.,Vincent,J.L.,Corbetta,M.,VanEssen,D.C.,&Raichle,M.E.
(2005).Thehumanbrainisintrinsicallyorganizedintodynamic,anticorrelated
ingwillbebolsteredbyadvancesintechnology.Developmentsin functional networks. Proceedings of the National Academy of Sciences of the
themethodsusedtoestimatemodelparameters,suchasBayesian UnitedStatesofAmerica,102(27),9673–9678.
estimation(e.g.,Turneretal.,2013),willallowresearcherstotest Glover,G.H.(1999).Deconvolutionofimpulseresponseinevent-relatedBOLD
fMRI.Neuroimage,9(4),416–429.
moresophisticatedmodelsthanarecurrentlypossible.Likewise,
Green,D.M.,&Swets,J.A.(1966).SignalDetectionTheoryandPsychophysics.New
advances in neuroimaging technologies, such as high-field fMRI York:Wiley.
(Duyn,2012),willallowformorefine-grainedmeasuresofhuman Harrison,S.A.,&Tong,F.(2009).Decodingrevealsthecontentsofvisualworking
memoryinearlyvisualareas.Nature,458(7238),632–635.
brainactivity.Spurredbyprogressonmultiplefronts,thefuture
Haxby,J.V.,Gobbini,M.I.,Furey,M.L.,Ishai,A.,Schouten,J.L.,&Pietrini,P.(2001).
ofmodel-basedneuroscienceholdsgreatpromiseforlinkingthe- Distributedandoverlappingrepresentationsoffacesandobjectsinventral
oreticalprocesseswiththeirunderlyingneuralinstantiations. temporalcortex.Science,293(5539),2425–2430.
Hayden, B. Y., Heilbronner, S. R., Pearson, J. M., & Platt, M. L. (2011). Surprise
signalsinanteriorcingulatecortex:neuronalencodingofunsignedreward
Acknowledgments prediction errors driving adjustment in behavior. Journal of Neuroscience,
31(11),4178–4187.
Haynes,J.D.,&Rees,G.(2005).Predictingthestreamofconsciousnessfromactivity
This research was supported by National Science Foundation inhumanvisualcortex.CurrentBiology,15(14),1301–1307.
grant BCS-1228526 to FT and National Institutes of Health Henson,R.(2005).Amini-reviewoffMRIstudiesofhumanmedialtemporallobe
activityassociatedwithrecognitionmemory.QuarterlyJournalofExperimental
FellowshipF32-EY022569toMP,andwasfacilitatedbyNational
Psychology Section B. Comparative and Physiological Psychology, 58(3–4),
InstitutesofHealthcentergrantP30-EY-008126totheVanderbilt 340–360.
VisionResearchCenter. Herrmann,K.,Heeger,D.J.,&Carrasco,M.(2012).Feature-basedattentionenhances
performancebyincreasingresponsegain.VisionResearch,74,10–20.
Honey,C.J.,Sporns,O.,Cammoun,L.,Gigandet,X.,Thiran,J.P.,Meuli,R.,etal.
References (2009).Predictinghumanresting-statefunctionalconnectivityfromstructural
connectivity.ProceedingsoftheNationalAcademyofSciencesoftheUnitedStates
ofAmerica,106(6),2035–2040.
Ashby,F.G.,&Maddox,W.T.(1993).Relationsbetweenprototype,exemplar,and Howard,M.W.,&Kahana,M.J.(2002).Adistributedrepresentationoftemporal
decisionboundmodelsofcategorization.JournalofMathematicalPsychology, context.JournalofMathematicalPsychology,46,269–299.
37(3),372–400. Huth,A.G.,Nishimoto,S.,Vu,A.T.,&Gallant,J.L.(2012).Acontinuoussemantic
Ashby,F.G.,&Maddox,W.T.(2005).Humancategorylearning.AnnualReviewof spacedescribestherepresentationofthousandsofobjectandactioncategories
Psychology,56,149–178. acrossthehumanbrain.Neuron,76(6),1210–1224.
Badre,D.,&Frank,M.J.(2012).Mechanismsofhierarchicalreinforcementlearning Ide,J.S.,Shenoy,P.,Yu,A.J.,&Li,C.S.(2013).Bayesianpredictionandevaluationin
in cortico-striatal circuits 2: evidence from fMRI. Cerebral Cortex, 22(3), theanteriorcingulatecortex.JournalofNeuroscience,33(5),2039–2047.
527–536. Kamitani,Y.,&Tong,F.(2005).Decodingthevisualandsubjectivecontentsofthe
Bonin,V.,Mante,V.,&Carandini,M.(2005).Thesuppressivefieldofneuronsin humanbrain.NatureNeuroscience,8(5),679–685.
lateralgeniculatenucleus.JournalofNeuroscience,25(47),10844–10856.
Kanwisher,N.,McDermott,J.,&Chun,M.M.(1997).Thefusiformfacearea:a
Borst,J.P.,&Anderson,J.R.(2014).UsingtheACT-Rcognitivearchitecturein moduleinhumanextrastriatecortexspecializedforfaceperception.Journal
combinationwithfMRIdata.InB.U.Forstmann,&E.J.Wagenmakers(Eds.), ofNeuroscience,17(11),4302–4311.
AnIntroductiontoModel-BasedCognitiveNeuroscience.NewYork:Springer. Kastner,S.,Pinsk,M.A.,DeWeerd,P.,Desimone,R.,&Ungerleider,L.G.(1999).
Botvinick,M.M.,Braver,T.S.,Barch,D.M.,Carter,C.S.,&Cohen,J.D.(2001).Conflict Increased activity in human visual cortex during directed attention in the
monitoringandcognitivecontrol.PsychologicalReview,108(3),624–652. absenceofvisualstimulation.Neuron,22(4),751–761.
Botvinick,M.M.,Cohen,J.D.,&Carter,C.S.(2004).Conflictmonitoringandanterior Kay,K.N.,Naselaris,T.,Prenger,R.J.,&Gallant,J.L.(2008).Identifyingnatural
cingulatecortex:anupdate.TrendsinCognitiveSciences,8(12),539–546. imagesfromhumanbrainactivity.Nature,452(7185),352–355.
Boynton,G.M.,Engel,S.A.,Glover,G.H.,&Heeger,D.J.(1996).Linearsystems Kintsch, W. (1967). Memory and decision aspects of recognition learning.
analysisoffunctionalmagneticresonanceimaginginhumanV1.Journalof PsychologicalReview,74(6),496–504.
Neuroscience,16(13),4207–4221. Kragel,J.E.,Morton,N.W.,&Polyn,S.M.(2015).Neuralactivityinthemedial
Brouwer, G. J., & Heeger, D. J. (2009). Decoding and reconstructing color temporalloberevealsthefidelityofmentaltimetravel.JournalofNeuroscience,
from responses in human visual cortex. Journal of Neuroscience, 29(44), 35(7),2914–2926.
13992–14003.

M.S.Pratte,F.Tong/JournalofMathematicalPsychology76(2017)80–93 93
Kriegeskorte,N.,Mur,M.,Ruff,D.A.,Kiani,R.,Bodurka,J.,Esteky,H.,etal.(2008). Polyn,S.M.,Norman,K.A.,&Kahana,M.J.(2009).Acontextmaintenanceand
Matchingcategoricalobjectrepresentationsininferiortemporalcortexofman retrievalmodeloforganizationalprocessesinfreerecall.PsychologicalReview,
andmonkey.Neuron,60(6),1126–1141. 116(1),129–156.
Logan,G.D.,Cowan,W.B.,&Davis,K.A.(1984).Ontheabilitytoinhibitsimpleand Pratte,M.S.,Ling,S.,Swisher,J.D.,&Tong,F.(2013).Howattentionextractsobjects
choicereactiontimeresponses:amodelandamethod.JournalofExperimental fromnoise.JournalofNeurophysiology,110(6),1346–1356.
Psychology:HumanPerceptionandPerformance,10(2),276–291. Raaijmakers,J.G.W.,&Shiffrin,R.M.(2002).Modelsofmemory.InH.Pashler,&
Love,B.C.(2015).Thealgorithmiclevelisthebridgebetweencomputationand D.Medin(Eds.),MemoryandCognitiveProcesses:Vol.2.Stevens’Handbookof
brain.TopicsinCognitiveScience,7(2),230–242. ExperimentalPsychology(thirded.).NewYork:JohnWiley&Sons,Inc.
Lu,Z.L.,&Dosher,B.A.(1998).Externalnoisedistinguishesattentionmechanisms. Ratcliff,R.,&Rouder,J.N.(1998).Modelingresponsetimesfortwo-choicedecisions.
VisionResearch,38(9),1183–1198. PsychologicalScience,9(5),347–356.
Lu,Z.L.,Li,X.,Tjan,B.S.,Dosher,B.A.,&Chu,W.(2011).Attentionextractssignal Rouder,J.N.,&Ratcliff,R.(2004).Comparingcategorizationmodels.Journalof
ExperimentalPsychology:General,133(1),63–82.
inexternalnoise:aBOLDfMRIstudy.JournalofCognitiveNeuroscience,23(5),
Rushworth,M.F.,Walton,M.E.,Kennerley,S.W.,&Bannerman,D.M.(2004).Action
1148–1159.
Mack,M.L.,Preston,A.R.,&Love,B.C.(2013).Decodingthebrain’salgorithm setsanddecisionsinthemedialfrontalcortex.TrendsinCognitiveSciences,8(9),
410–417.
for categorization from its neural implementation. Current Biology, 23(20),
Serences,J.T.,&Boynton,G.M.(2007).Feature-basedattentionalmodulationsin
2023–2027. theabsenceofdirectvisualstimulation.Neuron,55(2),301–312.
Marois,R.(2016).Thebrainmechanismsofworkingmemory:Anevolvingstory.
Serences,J.T.,&Saproo,S.(2009).Populationresponseprofilesinearlyvisualcortex
InP.Jolicoeur,C.Lefebvre,&J.Martinez-Trujillo(Eds.),MechanismsofSensory
arebiasedinfavorofmorevaluablestimuli.JournalofNeurophysiology,104(1),
WorkingMemory:AttentionandPerfomance(XXV).Elsevier.
76–87.
Marr,D.(1982).Vision.AComputationalInvestigationintotheHumanRepresentation Sereno,M.I.,Dale,A.M.,Reppas,J.B.,Kwong,K.K.,Belliveau,J.W.,Brady,T.J.,
andProcessingofVisualInformation.NewYork:Freeman. etal.(1995).Bordersofmultiplevisualareasinhumansrevealedbyfunctional
McClelland,J.L.,&Elman,J.L.(1986).TheTRACEmodelofspeechperception. magneticresonanceimaging.Science,268(5212),889–893.
CognitivePsychology,18(1),1–86. Shenoy,P.,Angela,J.Y.,&Rao,R.P.2010.Arationaldecisionmakingframework
McClelland,J.L.,&Rogers,T.T.(2003).Theparalleldistributedprocessingapproach forinhibitorycontrol.PaperpresentedattheAdvancesinneuralinformation
tosemanticcognition.NatureReviewsNeuroscience,4(4),310–322. processingsystems.
Mitchell,T.M.,Shinkareva,S.V.,Carlson,A.,Chang,K.M.,Malave,V.L.,Mason,R. Shenoy, P., & Yu, A. J. (2011). Rational decision-making in inhibitory control.
A.,etal.(2008).Predictinghumanbrainactivityassociatedwiththemeanings FrontiersinHumanNeuroscience,5,48.
ofnouns.Science,320(5880),1191–1195. Shepard,R.N.(1987).Towardauniversallawofgeneralizationforpsychological
Moradi,F.,&Heeger,D.J.(2009).Inter-ocularcontrastnormalizationinhuman science.Science,237(4820),1317–1323.
visualcortex.JournalofVision,9(3),11–22.13. Sprague,T.C.,Ester,E.F.,&Serences,J.T.(2014).Reconstructionsofinformation
Nichols,T.,&Hayasaka,S.(2003).Controllingthefamilywiseerrorrateinfunctional invisualspatialworkingmemorydegradewithmemoryload.CurrentBiology,
neuroimaging:acomparativereview.StatisticalMethodsinMedicalResearch, 24(18),2174–2180.
12(5),419–446. Stone,M.(1960).Modelsforchoice-reactiontime.Psychometrika,25(3),251–260.
Norman,K.A.,Polyn,S.M.,Detre,G.J.,&Haxby,J.V.(2006).Beyondmind-reading: Tavor, I., Parker Jones, O., Mars, R. B., Smith, S. M., Behrens, T. E., & Jbabdi, S.
(2016).Task-freeMRIpredictsindividualdifferencesinbrainactivityduring
multi-voxelpatternanalysisoffMRIdata.TrendsinCognitiveSciences,10(9),
taskperformance.Science,352(6282),216–220.
424–430.
Nosofsky,R.M.,&Palmeri,T.J.(1997).Anexemplar-basedrandomwalkmodelof Tong,F.,&Pratte,M.S.(2012).Decodingpatternsofhumanbrainactivity.Annual
ReviewofPsychology,63,483–509.
speededclassification.PsychologicalReview,104(2),266–300.
Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.J.,Brown,S.D.,Sederberg,P.B.,&
O’Doherty,J.P.,Dayan,P.,Friston,K.,Critchley,H.,&Dolan,R.J.(2003).Temporal
Steyvers,M.(2013).ABayesianframeworkforsimultaneouslymodelingneural
differencemodelsandreward-relatedlearninginthehumanbrain.Neuron,
andbehavioraldata.Neuroimage,72,193–206.
38(2),329–337.
Turner,B.M.,vanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
O’Doherty,J.P.,Dayan,P.,Schultz,J.,Deichmann,R.,Friston,K.,&Dolan,R.J.(2004).
stractionsthroughneuroimaging:theneuraldriftdiffusionmodel.Psychologi-
Dissociablerolesofventralanddorsalstriatumininstrumentalconditioning.
calReview,122(2),312–336.
Science,304(5669),452–454. Verbruggen, F., & Logan, G. D. (2008). Response inhibition in the stop-signal
O’Doherty,J.P.,Hampton,A.,&Kim,H.(2007).Model-basedfMRIanditsapplication paradigm.TrendsinCognitiveSciences,12(11),418–424.
torewardlearninganddecisionmaking.AnnalsoftheNewYorkAcademyof Wagner,A.D.,Schacter,D.L.,Rotte,M.,Koutstaal,W.,Maril,A.,Dale,A.M.,etal.
Sciences,1104,35–53. (1998).Buildingmemories:rememberingandforgettingofverbalexperiences
Polyn,S.M.,Natu,V.S.,Cohen,J.D.,&Norman,K.A.(2005).Category-specific aspredictedbybrainactivity.Science,281(5380),1188–1191.
corticalactivityprecedesretrievalduringmemorysearch.Science,310(5756), Xu,Y.,&Chun,M.M.(2006).Dissociableneuralmechanismssupportingvisual
1963–1966. short-termmemoryforobjects.Nature,440(7080),91–95.