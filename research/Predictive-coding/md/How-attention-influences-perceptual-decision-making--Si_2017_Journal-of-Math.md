JournalofMathematicalPsychology76(2017)117–130
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
How attention influences perceptual decision making: Single-trial
EEG correlates of drift-diffusion model parameters
MichaelD.Nuneza,c,∗ ,JoachimVandekerckhovea,c,d,RameshSrinivasana,b,d
aDepartmentofCognitiveSciences,UniversityofCalifornia,Irvine,CA,USA
bDepartmentofBiomedicalEngineering,UniversityofCalifornia,Irvine,CA,USA
cDepartmentofStatistics,UniversityofCalifornia,Irvine,CA,USA
dInstituteofMathematicalBehavioralSciences,UniversityofCalifornia,Irvine,CA,USA
h i g h l i g h t s
• Ourgoalistoelucidatetheeffectofattentiononvisualdecisionmaking.
• ModelswithlinearrelationshipsbetweendiffusionmodelparametersandEEGwerefit.
• Single-trialEEGattentionmeasuresexplainevidenceaccumulationandpreprocessing.
• Accuracyandcorrectreactiontimedistributionswerepredictedout-of-sample.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Perceptual decision making can be accounted for by drift-diffusion models, a class of decision-
Availableonline11April2016 makingmodelsthatassumeastochasticaccumulationofevidenceoneachtrial.Fittingresponsetime
and accuracy to a drift-diffusion model produces evidence accumulation rate and non-decision time
Keywords:
parameterestimatesthatreflectcognitiveprocesses.Ourgoalistoelucidatetheeffectofattentionon
Visualattention
visualdecisionmaking.Inthisstudy,weshowthatmeasuresofattentionobtainedfromsimultaneousEEG
Perceptualdecisionmaking
recordingscanexplainper-trialevidenceaccumulationratesandperceptualpreprocessingtimesduring
Diffusionmodels
Neurocognitivemodeling avisualdecisionmakingtask.Modelsassuminglinearrelationshipsbetweendiffusionmodelparameters
Electroencephalography(EEG) andEEGmeasuresasexternalinputswerefitinasinglestepinahierarchicalBayesianframework.The
HierarchicalBayesianmodeling EEGmeasureswerefeaturesoftheevokedpotential(EP)totheonsetofamaskingnoiseandtheonsetof
atask-relevantsignalstimulus.Single-trialevokedEEGresponses,P200stotheonsetsofvisualnoiseand
N200stotheonsetsofvisualsignal,explainsingle-trialevidenceaccumulationandpreprocessingtimes.
Within-trialevidenceaccumulationvariancewasnotfoundtobeinfluencedbyattentiontothesignalor
noise.Single-trialmeasuresofattentionleadtobetterout-of-samplepredictionsofaccuracyandcorrect
reactiontimedistributionsforindividualsubjects.
©2016ElsevierInc.Allrightsreserved.
1. Introduction thedrivercansuppresstheirattentiontothegreenarrowand/or
attendtothecorrectredlightintheirlane.Thedecisiontostopor
Therearemanysituationsontheroadwhenthedriverofave- accelerateisanexampleofaperceptualdecision.Perceptualdeci-
hiclemustdecidetostoporacceleratethroughanintersectionby sionmakingistheprocessofmakingquickdecisionsbasedonob-
observingatrafficlight.Thepresenceofagreenarrowforanad- jects’featuresobservedwiththesenses.Asshowninthestoplight
jacentlane(i.e.thedistractoror‘‘noise’’)canbedistractingforthe example,attentionishighlyinfluentialintheperceptualdecision
driverwhoselightisred(i.e.the‘‘signal’’).Thepresenceofthedis- makingprocess.Whendistractingobjectsexistinvisualspace,one
tractoraffectsthereactiontimeandchoiceofthedriver.However mustattendonlytotherelevantobjectsandactivelyignoredis-
tractingobjects.Eachtimeanindividualreachesastoplight,they
willbemorelikelytomakeasaferdecisioniftheysuppressdis-
∗ tractingvisualinputandenhancerelevantvisualinput.
Corresponding author at: Department of Cognitive Sciences, University of
California,Irvine,CA,USA. Thegoalofthisstudywastoevaluatewhetherattentioncould
E-mailaddress:mdnunez1@uci.edu(M.D.Nunez). predict different components of the decision making process on
http://dx.doi.org/10.1016/j.jmp.2016.03.003
0022-2496/©2016ElsevierInc.Allrightsreserved.

118 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
eachtrialofavisualdiscriminationexperiment.Wemakeuseof (fMRI) and EEG recordings into hierarchical models of choice-
high-density electroencephalographic (EEG) recordings from the RT (e.g. Mulder, Van Maanen, & Forstmann, 2014; Nunez et al.,
human scalp to find single-trial evoked potentials (EPs) to the 2015;Turner,vanMaanen,&Forstmann,2015).Inthisstudy,we
useahierarchicalformofthediffusionmodel(Vandekerckhove,
| onset of | visual | signal and | to the onset | of a | distractor | (mask) to |     |     |     |     |     |     |     |     |
| -------- | ------ | ---------- | ------------ | ---- | ---------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
measure the deployment of attention to task-relevant features. Tuerlinckx,&Lee,2011),allowingvariabilitybetweenparticipants
Wefoundthatoneachtrial,modulationsoftheevokedpotentials andacrossconditions,topredictanddescribesingle-trialreaction
by attention were predictive of specific components of a drift- times and accuracy during a visual decision making task. While
diffusionmodelofthedecisionmakingprocess. other similar models of choice-RT have successfully predicted
behaviorduringvisualdecisionmaking,suchasthesimplerlinear
ballisticaccumulatormodel(Brown&Heathcote,2008)oramore
1.1. Visualattentionanddecisionmaking
|           |     |            |              |        |         |          | complicated    | drift-diffusion |     | model         | that | intrinsically | accounts | for        |
| --------- | --- | ---------- | ------------ | ------ | ------- | -------- | -------------- | --------------- | --- | ------------- | ---- | ------------- | -------- | ---------- |
|           |     |            |              |        |         |          | trial-to-trial | variability     |     | in parameters |      | within        | subjects | (Ratcliff, |
| Attention | is  | beneficial | for decision | making | because | relevant |                |                 |     |               |      |               |          |            |
1978;Ratcliff&McKoon,2008),wehavechosenadiffusionmodel
| features | of the | environment | can be preferentially |     | processed | to  |     |     |     |     |     |     |     |     |
| -------- | ------ | ----------- | --------------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thatallowsustotestspecificpredictionsfrommodelsofattention
| enhance | the quality | of evidence. | During | visual | tasks individuals |     |     |     |     |     |     |     |     |     |
| ------- | ----------- | ------------ | ------ | ------ | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(i.e.Lu&Dosher,1998;Smith&Ratcliff,2009)whilebeingsimple-
maydeploydifferentattentionstrategiessuchas:enhancingthe
|                     |     |          |                      |     |                |     | enough | to fit in | reasonable | time | periods | given | the hierarchical |     |
| ------------------- | --- | -------- | -------------------- | --- | -------------- | --- | ------ | --------- | ---------- | ---- | ------- | ----- | ---------------- | --- |
| signal, suppressing |     | external | noise (distractors), |     | or suppressing |     | form.  |           |            |      |         |       |                  |     |
internal noise (Dosher & Lu, 2000; Lu & Dosher, 1998). These In the drift-diffusion model it is assumed that subjects accu-
strategiesarethoughttochangebasedonthesignaltonoiseratio mulateevidenceforonechoiceoveranother(oracorrectversus
of the stimulus, such that individuals will enhance sensory gain incorrect response, as in this study) in a random walk evidence
tobothsignalandnoiseduringperiodsoflownoiseandsharpen
accumulationprocesswithaninfinitesimaltimestep.Thatis,ev-
attentiontoonlysignalduringperiodsofhighnoise(Lu&Dosher,
idenceE t accumulatesfollowingaWienerprocess(i.e.Brownian
1998), although specific strategies have been shown to differ δ ς2
|                 |     |            |                   |     |               |       | motion) | with drift | rate | and | instantaneous |     | variance | (Ross, |
| --------------- | --- | ---------- | ----------------- | --- | ------------- | ----- | ------- | ---------- | ---- | --- | ------------- | --- | -------- | ------ |
| across subjects |     | (Bridwell, | Hecker, Serences, |     | & Srinivasan, | 2013; |         |            |      |     |               |     |          |        |
2014)suchthat
Krishnan,Kang,Sperling,&Srinivasan,2013;Nunez,Srinivasan,&
| Vandekerckhove,2015).Multiplegroupshaveproposedmodelsof |     |     |     |     |     |     |     |     |     | dE         |     |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
|                                                         |     |     |     |     |     |     |     |     |     | t ∼N(δ,ς2) |     |     |     | (1) |
visual attention and decision making that yield diverse reaction dt
δ
time and choice distributions dependent upon attentional load Thusthedriftrate describesmeanrateofevidenceaccumu-
(Smith&Ratcliff,2009;Spieler,Balota,&Faust,2000).Attention lationwithinatrialandthediffusioncoefficient ς influencesthe
canbedeployedtothefeaturesand/orlocationofastimulus,and varianceofevidenceaccumulationwithinonetrial,withthetrue
attentioncanbenefitdecisionmakingwhenthesubjectiscuedto varianceofthecurrentevidenceatanyparticulartimetbeing ς2t.
the location or features of the stimulus (Davis & Graham, 1981; Agraphicalrepresentationofthediffusionmodelisprovidedinthe
| Eriksen&Hoffman,1972;Shaw&Shaw,1977). |     |     |     |     |     |     | middlepanelofFig.1. |     |     |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- |
Event-related potentials (ERPs) are trial-averaged EEG re- A few other parameters describe a diffusion model. The
sponsestoexternalstimuli.VisualERPs(alsolabeledVisualEvoked boundaryseparation α isequaltotheamountofrelativeevidence
Potentials; VEPs) have been shown to track visual attention to requiredtomakechoiceAoverchoiceB(ormakeacorrectdecision
theonsetofstimuli(Harter&Aine,1984;Luck,Woodman,&Vo- overanincorrectdecision),andtheboundaryseparationhasbeen
gel,2000).Thatis,amplitudesofthepeaksoftheERPwaveform showntobemanipulatedbyspeedvs.accuracystrategytrade-offs
(i.e., ERP ‘‘components’’) within certain millisecond-scale time (Voss,Rothermund,&Voss,2004).Theparameterthatencodesthe
β
windows are shown to be larger when subjects encounter task- startingpositionofevidence reflectsthebiastowardonechoice
relevantstimuliintheexpectedlocationinvisualspace.Twocom- oranother(equalto0.5whenmodelingcorrectversusincorrect
τ
ponentsofparticularinterestaretheN1(orN200)andP2(orP200) choicesasinthisstudy).Thenon-decisiontime isequaltothe
components.ThetermsN1andP2refertheorderofnegativeand amountoftimewithinthereactiontimeofeachtrialthatisnot
positivepeaksinthetimeseriesrespectively,andthemoregeneral dedicatedtothedecisionmakingprocess.Typicallynon-decision
alternativenamesN200andP200refertotheirapproximatelaten- time is assumed to be equal to the sum of preprocessing time
|     |     |     |     |     |     |     | before the | evidence | accumulation |     | process | and | motor | response |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ------------ | --- | ------- | --- | ----- | -------- |
ciesinmilliseconds.ChangesinN200latencieshavebeenshown
timeafteradecisionhasbeenreached.Therelativecontribution
tocorrelatewithattentionalload(Callaway&Halliday,1982),and
ofthesetwonon-decisiontimesisnotidentifiablefrombehavior
N200measureshaveevenbeenusedinBrain–ComputerInterfaces
aloneandthereforeisrarelyexplicitlymodeled.
(BCI)thatmakeuseofsubjects’attentiontospecificchangingstim-
|           |            |          |                |      |           |        | All three                                               | of the | parameters |     | related | to evidence | accumulation |        |
| --------- | ---------- | -------- | -------------- | ---- | --------- | ------ | ------------------------------------------------------- | ------ | ---------- | --- | ------- | ----------- | ------------ | ------ |
| uli, such | as letters | in a BCI | speller (Hong, | Guo, | Liu, Gao, | & Gao, |                                                         |        |            |     |         |             |              |        |
|           |            |          |                |      |           |        | arenotidentifiablewithbehavioraldataalone(i.e.driftrate |        |            |     |         |             |              | δ ,the |
2009).Findingsinthesetrial-averagedEEG(ERP)studiessuggest
|                  |     |                   |                  |     |        |          | diffusioncoefficient |     | ς ,andtheboundaryseparationa).Onlytwo |     |     |     |     |     |
| ---------------- | --- | ----------------- | ---------------- | --- | ------ | -------- | -------------------- | --- | ------------------------------------- | --- | --- | --- | --- | --- |
| that information |     | is also available | in single-trials |     | of EEG | that can |                      |     |                                       |     |     |     |     |     |
beusedtoevaluatetherelationshipbetweenattentionanddeci- of the three parameters can be assumed to vary across subjects
|     |     |     |     |     |     |     |     |     |     | ς   |     |     |     | δ   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sionmaking.InthispaperwewillusethealternativenamesP200 andtrials(e.g.multiplying bytwoanddividingbothaand by
twowouldresultinthesamefitofchoice-RT)(Ratcliff&McKoon,
andN200because(1)theexacttimewindowsofcomponentsvary
2008;Wabersich&Vandekerckhove,2014).Previousstudieshave
| acrossstudies,(2)componentsinthisstudywerebothlocalizedto |     |     |     |     |     |     |                                             |     |     |     |     |     | ς             |     |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | ------------- | --- |
|                                                           |     |     |     |     |     |     | typicallychosentofixthediffusioncoefficient |     |     |     |     |     | to1or0.1(Van- |     |
around200ms,and(3)componentsinthisstudywerefoundon
|     |     |     |     |     |     |     | dekerckhove | et  | al., 2011; | Wabersich |     | & Vandekerckhove, |     | 2014). |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ---------- | --------- | --- | ----------------- | --- | ------ |
single-trialsasopposedtointhetrial-average.
|     |     |     |     |     |     |     | However | due to | the predictions |     | made | by Dosher | and | Lu (2000), |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | --------------- | --- | ---- | --------- | --- | ---------- |
inthatinternalnoiseissuppressedbyattentiontothesignal,we
|     |     |     |     |     |     |     |     | ς   |     |     |     |     | α   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1.2. Behavioralmodelsofdecisionmaking choosetoleave tovary.Theboundaryseparation wasfixedat
1foralltrialsandsubjects.Ourprimaryanalysisfocusedonthe
Drift-diffusionmodelsareawidely-usedclassofmodelsused trial-to-trialvariabilityintheevidenceaccumulationprocessdue
tojointlypredictsubjects’choicesandreactiontimes(RT)during tofluctuationsinattentionfromtrial-to-trialwithinindividuals.
two-choice decision making (Ratcliff, 1978; Ratcliff & McKoon, Although trial-to-trial speed–accuracy trade-offs can be experi-
2008; Stone, 1960). ‘‘Neural’’ drift-diffusion models have also mentallyintroducedtofindneuralcorrelatesoftheboundarysep-
successfullyincorporatedfunctionalmagneticresonanceimaging aration(e.g.vanMaanenetal.,2011)ormayexistduetoper-trial

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 119
Fig.1. TwotrialsofSubject10’sSVDweightedEEG(TopandBottomwithbounds−85to85µV)andrepresentationsofthissubject’sevidenceaccumulationprocesson6
lownoisetrials(Middle).Evidenceforacorrectresponseinoneexampletrial(denotedbytheredline)firstremainsneutralduringaninitialperiodofvisualpreprocessing
timeτ pre.Thenevidenceisaccumulatedwithaninstantaneousevidenceaccumulationrateofmeanδ(thedriftrate)andstandarddeviationς(thediffusioncoefficient)viaa
Wienerprocess.Thesubjectacquireseitherα=1evidenceunitor0evidenceunitstomakeacorrectorincorrectdecisionrespectively.Afterenoughevidenceisreachedfor
eitherdecision,motorresponsetimeτ motorexplainstheremainderofthattrial’sobservedreactiontime.The85thand15thpercentilesofSubject10’ssingle-trialdriftrates
δ i,10,1inthelownoiseconditionareshownasorangeandgreenvectors,suchthatitwouldtake253and299msrespectivelytoaccumulatethe0.5evidenceaccumulation
unitsneedtomakeacorrectdecisioniftherewasnovarianceintheaccumulationprocess.Thelargerdriftrateisalinearfunctionofthelargersingle-trialN200amplitude
(**),whilethesmallerdriftrateisalinearfunctionofthesmallerN200amplitude(*).Thescalpactivation(SVDweightsmultipliedbyonetrial’sN200amplitude)ofthis
subject’sresponsetothevisualsignalrangesfrom−13to13µVonbothtrials.Thetwodarkblueandredevidencetimecourseswererandomlygeneratedtrialswiththe
largerdriftrate.Thethreedotted,lightblueevidencetimecourseswererandomlygeneratedtrialswiththesmallerdriftrate.TrueWienerprocesseswithdriftsδ i,10,1and
diffusioncoefficientς 10,1wereestimatedusingasimplenumericaltechniquediscussedinBrownetal.(2006).(Forinterpretationofthereferencestocolorinthisfigure
legend,thereaderisreferredtothewebversionofthisarticle.)
performancefeedback(Dutilhetal.,2012),wehavenoreasonto 1.4. Hypothesizedattentioneffects
believethattheboundaryseparationwillvaryconsiderablyfrom
trial-to-trialwithinasubjectduetochangesinattention.Moreover, Anintegratedmodelofvisualattention,visualshorttermmem-
allsubjectsweregiventhesameaccuracyinstructiontomaintain ory,andperceptualdecisionmakingbySmithandRatcliff(2009)
similaraccuracyvs.speedtrade-offsacrosssubjects. predictsthatattentionoperatesontheencodingofthestimulus,
andthatenhancedencodingincreasesdriftrateduringthedeci-
1.3. Single-trialEEGmeasuresofattention sionmakingprocess.Furthermore,themodelpredictsthatvisual
encodingtime(i.e.visualpreprocessing)willbereducedbyatten-
EEG correlates of attention and decision making have been tionwhichisreflectedinthenon-decisiontimeparameter.How-
foundusingclassificationmethods.Onegrouphasshownthatthe ever, this model of visual attention only considers the detection
amplitude of certain EEG components in the time domain track ofastimulusinanotherwiseblankfield—thatis,afieldwithno
typeanddurationoftwo-alternativeforcedchoiceresponsesand visualnoise.Thus,itdoesnothavepredictionsforthedistinctpro-
thenshowedthatthesecomponents’amplitudestrackedevidence cessesofnoisesuppressionandsignalenhancement,asinthePer-
accumulationrate(Philiastides&Sajda,2006;Ratcliff,Philiastides, ceptualTemplateModel(Lu&Dosher,1998).Signalenhancement
&Sajda,2009).HowevertheEEGcomponentsinthesestudieswere duringtheevidenceaccumulationprocessispredictedtoreduce
foundbyfindingthemaximumpredictorsofthebehavioraldata thediffusioncoefficient ς becausethemechanismbywhichsig-
andthushadnoaprioriinterpretation.Anothergrouphasfound nalenhancementtakesplace,accordingtothePerceptualTemplate
that single-trial amplitude in a few frequency bands, especially Model,isadditiveinternalnoisereduction1 (Dosher&Lu,2000);
the 4–9 Hz theta band, predicts evidence accumulation rates thismechanismispredictedtobemosteffectiveinlownoisecon-
(van Vugt, Simen, Nystrom, Holmes, & Cohen, 2012). However ditionssincedecreasinginternalnoisewillleadtobetterprocess-
theseoscillationswerefoundusingcanonicalcorrelationanalysis ing of both the visual signal and external visual noise. External
(CCA; Calhoun, Adali, Pearlson, & Pekar, 2001), a data driven noisesuppression,ontheotherhand,isexpectedtoreflecttheen-
algorithm that found any EEG channel mixtures that contained codingofthestimulusbymanipulationofaperceptualtemplate,
correlationswiththedriftdiffusionmodelparameters.Whilethe increasingtheaveragerateofevidenceaccumulation δ byimprov-
results were confirmed using cross validation, the set of EEG ingtheoverallqualityofevidenceonatrial.ThePerceptualTem-
identifiedbythismethodalsodidnothaveanaprioriexplanation. plateModelpredictsthismechanismismosteffectiveinhighnoise
These studies point us in directions of exploration and perform conditions.
wellatprediction,butwehavelittleinformationastowhetherthe
EEGinformationreflectedattention,thedecisionprocessitself,or
someothercorrelateofevidenceaccumulation.Inthisstudy,we
introduceasimpleprocedurethatisinformedbyERPsknowntobe 1 ‘‘In stimulus enhancement, attention increases the gain on the stimulus,
whichisformallyequivalenttoreducinginternaladditivenoise.Thiscanimprove
relatedtoattention,andwemakeuseofsingle-trialERPestimates
performanceonlyinlowexternalnoisestimuli,sinceexternalnoiseisthelimiting
tomodelbehavioronsingletrials.
factorinhighexternalnoisestimuli.’’(Dosher&Lu,2000,p.1272).

120 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
Fig.2. Thetimecourseofonetrialoftheexperimentalstimulus.Onetrialconsistedofthefollowing:(1)750msoffixationonablackcrossonagrayscreen,then(2)visual
contrastnoisechangingat8Hzfor750mswhilemaintainingfixation(dubbedthecueinterval)and(3)acircularfieldofsmallorientedbarsflickeringat15Hzoverlaid
onthechangingvisualnoisefor1000–2000mswhilemaintainingfixation(dubbedtheresponseinterval).Thesubjects’taskwastoindicateduringtheresponseinterval
whetherthebarswereonaverageorientedtowardthe‘‘top-right’’(45°fromhorizontal;asinthisexample)orthe‘‘top-left’’(135°fromhorizontal).Itwasassumedthat
subjects’decisionmakingprocessoccurredonlyduringtheresponseintervalbutcouldbeinfluencedbybothonsetattentiontothevisualnoiseduringthecueintervaland
onsetattentiontovisualsignalduringtheresponseinterval.
In a previous study we showed that individual differences in wereusefulinthepreviousstudybutwillnotbeusedinthisstudy.
noisesuppressionpredictsindividualdifferencesinevidenceac- StimuliwerebuiltanddisplayedusingtheMATLABPsychophysics
cumulationratesandnon-decisiontimes(Nunezetal.,2015).We toolbox(Psychtoolbox-2;www.psychtoolbox.org).
alsoshowedthatdifferencesacrossindividualsinsignalenhance- Subjects viewed each trial of the experimental stimulus on
mentpredictindividualdifferencesinnon-decisiontimesandevi- a monitor in a dark room. Subjects sat 57 cm away from the
denceaccumulationvariance(i.e.thediffusioncoefficient),which monitor. The entire circular field of small oriented bars was
weassumetracksinternalnoiseinthesubject.Thefindingsofsig- 9.5cmindiameter,correspondingtoavisualangleof9 . 5°.Within
nalenhancementeffectsonevidenceaccumulationvarianceand eachtrialsubjectsfirstobservedablackcrossfor750msinthe
noisesuppressioneffectsonevidenceaccumulationrateseemto centerofthescreen,onwhichtheywereinstructedtomaintain
correspond closely to predictions made by the Perceptual Tem- fixationthroughoutthetrial.Subjectsthenobservedvisualnoise
plate Model. However the Perceptual Template Model does not for 750 ms. This time period of the stimulus will henceforth be
makeexplicitpredictionsaboutattentioneffectsonnon-decision referred to as cue interval, with the onset EEG response at the
times.Thepreviousstudydidnotexplorehowtrial-to-trialvari- beginning of this interval being the response to the noise (or
ationinattentionaffectedtrial-to-trialcognitivedifferences.Indi- ‘‘distractor’’)stimulus.Subjectsthenobservedthecircularfieldof
vidualdifferencesinattentioncouldbefoundthatarenotdetected smallorientedbarsoverlaidonthesquarefieldofvisualnoisefor
tobechangingwithinasubject,and/ortrial-to-trialvariabilityin 1000–2000msandrespondedduringthisinterval.Subjectswere
attentioncouldoccurthatdoesnotchangeacrossindividuals.In instructedtorespondasaccuratelyaspossiblewhileprovidinga
thisstudy,weshowthatwithin-subject,trial-to-trialvariabilityin responseduringeverytrial.Becauseevidencerequiredtomakea
attentiontobothnoiseandsignalpredictvariabilityindriftrate decisiononlyappearedduringthistimeframe,thedecisionprocess
andnon-decisiontimes,correspondingcloselytopredictionsmade was assumed to take place during this interval. This interval is
by the model of Smith and Ratcliff (2009) that predicts speeded referred to as the response interval, and attentional onset EEG
encoding time and increased evidence accumulation rate due to measuresduringthistimeperiodarereferredtoasresponsesto
enhancedattention.Thetwostudiestogethersuggestthatwithin- thesignalstimulus.Aftertheresponseintervalthefixationcross
ς
trialevidenceaccumulationvariances variedacrossindividuals, wasagainshownfor250mstoalertthesubjectsthatthetrialwas
butwedidnotfindevidencethatthismeasurevariedwithinindi- overandtocollectanydelayedresponses.
vidualsduetochangesintrial-to-trialattention. Threelevelsofvarianceofbarrotationandthreelevelsofnoise
luminance were used to modulate the task difficulty. However
2. Methods onlythenoiseluminancemanipulationisrelevantfortheanalysis
presentedhere.Averageluminanceofthenoisewas50%andthe
2.1. Experimentalstimulus:barfieldorientationtask luminance of the bars was 15%. In the low noise condition, that
luminance was drawn randomly at each pixel from a uniform
Reported in a previous study, behavioral and EEG data were distribution of 35% to 65% luminance. In the medium and high
collectedfromasimpletwo-alternativeforcedchoicetasktotest noise conditions, noise luminance was drawn randomly at each
individual differences in attention during visual decision making pixel from a uniform distribution of 27.5% to 72.5% and 20% to
(Nunezetal.,2015).Here,wereanalyzethesedatatoexploreper- 80% luminance respectively. Each subject experienced 180 trials
trial attention effects on the decision making process. Subjects fromeachnoisecondition,interleaved,foratotalof540trialssplit
wereinstructedoneachtrialtodifferentiatethemeanrotationof evenlyover6blocks.Thetotaldurationofthevisualexperiment
afieldofsmallbarsthatwereeitherorientedat45°or135°from foreachsubjectwasapproximatelyonehourand15minincluding
horizontal on average. Two representative frames of the display electivebreaksbetweenblocks.Moredetailsoftheexperimentcan
andthetimecourseofatrialareprovidedinFig.2.Thecircular befoundinourpreviouspublication(Nunezetal.,2015).
fieldofsmallbarswasembeddedinasquarefieldofvisualnoise Behavioral and EEG data were collected concurrently from
thatwaschangingat8Hz.Thebarfieldwasflickeringat15Hz. 17 subjects. Subjects performed accurately during the task. The
Thesefrequencieswerechosentoevokesteady-statevisualevoked across-subjectmean,standarddeviation,andmedianofaccuracy
potentials(SSVEPs),stimulusfrequency-taggedEEGresponsesthat were90 . 1%±5 . 8% , y = 91 . 6%,whiletheacross-subjectmean,

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 121
standard deviation, and median of average reaction time were ofthetrialsacrossallconditionsforeachsubjectineachwindow.
678±106 , t =670ms.Individualdifferencesinbehaviorexisted Thisrandomsetoftrialswasthesamesetusedasthetrainingset
acrosssubjectswiththemostaccuratesubjectanswering98.3%of forcrossvalidation,tobediscussedlater.Thetestsetsoftrialswere
trialscorrectlyandtheleastaccuratesubjectanswering78.5%of notusedtocalculatethetraditionalERPs.
trialscorrectly.Twodifferentsubjectswerethefastestandslowest Singular value decomposition (SVD; analogous to principal
withmeanRTsof502msand866msrespectively. componentanalysis)ofthetrialaverageswerethenusedtofind
linearmixturesofchannelsthatexplainedthelargestamountof
2.2. Single-trialEEGpredictors the variance in the ERP data (i.e. the first right-singular vectors
|     |     |     |     |     |     |     | v, explaining | a   | percentage | of  | variance | from | 39.4% | to 91.9% | and |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------- | --- | -------- | ---- | ----- | -------- | --- |
Electroencephalograms (EEG) were recorded using Electrical 45.0% to 93.2% across subjects in the cue and response intervals
|              |                     |     |             |          |            |            | respectively). | The    | first   | right-singular |           | vectors | were     | then used | as     |
| ------------ | ------------------- | --- | ----------- | -------- | ---------- | ---------- | -------------- | ------ | ------- | -------------- | --------- | ------- | -------- | --------- | ------ |
| Geodesics,   | Inc.’s high density |     | 128-channel | Geodesic |            | Sensor Net |                |        |         |                |           |         |          |           |        |
|              |                     |     |             |          |            |            | weights        | to mix | the raw | EEG            | data into | a brain | response |           | biased |
| and Advanced | Neuro Technology’s  |     | amplifier.  |          | Electrical | activity   |                |        |         |                |           |         |          |           |        |
towardthemaximumresponsetothevisualstimuli,yieldingone
fromthescalpwasrecordedatasamplingrateof1024samples
|            |                |         |           |     |       |          | time course | of       | the EEG        | per trial | for | both the   | cue       | and response |       |
| ---------- | -------------- | ------- | --------- | --- | ----- | -------- | ----------- | -------- | -------------- | --------- | --- | ---------- | --------- | ------------ | ----- |
| per second | with an online | average | reference |     | using | Advanced |             |          |                |           |     |            |           |              |       |
|            |                |         |           |     |       |          | intervals.  | A visual | representation |           | of  | the simple | procedure |              | for a |
NeuroTechnologysoftware.TheEEGdatawasthenimportedinto
MATLABforofflineanalysis.Lineartrendswereremovedfromthe singletrialisprovidedinFig.3.TherawdatamatrixEofdimension
|     |     |     |     |     |     |     | ×C  |     |     |     |     |     |     |     | ×1  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
EEGdata,andthedatawerebandpassfilteredtoa1–50Hzwindow N wasmultipliedbythefirstrightsingularvectorv(aC
vectorofchannelweights)toobtainaN×1vectorEv=e,which
usingahighpassButterworthfilter(1Hzpassbandwitha1dB
×
rippleanda0.25Hzstopbandwith10dBattenuation)andalow could then be split up into epochs of length T 1 representing
pass Butterworth filter (50 Hz pass band with 1 dB ripple and a theresponsetothestimulusoneachtrial.Notethatthevoltage
60Hzstopbandwith10dBattenuation). amplitudesoftheERPmeasurescalculatedbasedonthismethod
willdifferfromtraditionalsingleelectrodeERPamplitudessince
| EEG | artifact is broadly | defined | as data | collected |     | within EEG |     |     |     |     |     |     |     |     |     |
| --- | ------------------- | ------- | ------- | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thesingle-trialestimatesareaweightedsumofpotentialsoverall
recordingsthatdoesnotoriginatefromthebrain.Electricalartifact
electrodes.
canbebiological(e.g.fromthemuscles-EMGorfromthearteries-
NotonlydidthismethodboosttheSNRoftheEEGmeasures,
EKG)ornon-biological(e.g.temporaryelectrodedislocations,DC
|     |     |     |     |     |     |     | but this | method | also | reduced | EEG measures |     | of size | T × | C on |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------ | ---- | ------- | ------------ | --- | ------- | --- | ---- |
shifts,or60Hzlinenoise).Contributionofmuscleandelectrical
×
artifactwasreducedinrecordingsbyusinganextendedInfomax each trial to one latent variable that varies in time of size T
Independent Component Analysis algorithm (ICA; Lee, Girolami, 1. Thus the correlation of the EEG as inputs to the model was
& Sejnowski, 1999; Makeig, Bell, Jung, & Sejnowski, 1996). ICA drasticallyreducedandtheinterpretabilityofmodelparameters
algorithms are used to find linear mixtures of EEG data that wasincreasedcomparedtoanalyseswithhighlycorrelatedmodel
relatetospecificartifact.Componentsthatareindicativeofartifact inputs.Theweightvectorvforeachsubjectinboththecueand
typically have high spatial frequency scalp topographies, high response intervals also yields a scalp map when the values of
temporalfrequencyora1 / ffrequencyfalloff,andarepresenteither theweightsareinterpolatedbetweenelectrodes.Channelweights
calculatedusingSVDonsubject’sERPstothenoiseonset(during
| in only | a few trials or | intermittently | throughout |     | the | recording. |     |     |     |     |     |     |     |     |     |
| ------- | --------------- | -------------- | ---------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thecueinterval)areshownintopographicplotsforeachsubject
| These properties | are not | shared | by electrical |     | signals | from the |     |     |     |     |     |     |     |     |     |
| ---------------- | ------- | ------ | ------------- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inFig.7.ChannelweightscalculatedusingSVDonsubject’sERPs
brainasrecordedonthescalp(Nunez&Srinivasan,2006).Using
|                |            |          |        |     |            |          | to the signal | onset | (during | the | response | interval) |     | are shown | in  |
| -------------- | ---------- | -------- | ------ | --- | ---------- | -------- | ------------- | ----- | ------- | --- | -------- | --------- | --- | --------- | --- |
| these metrics, | components | manually | deemed |     | to reflect | artifact |               |       |         |     |          |           |     |           |     |
topographicplotsforeachsubjectinFig.6.WhilerawEEGonsingle
| were projected | into EEG | space | and subtracted |     | from | the raw |     |     |     |     |     |     |     |     |     |
| -------------- | -------- | ----- | -------------- | --- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
data. Components deemed to be a mixture of artifact and brain trials from single electrodes may have large enough SNRs to be
activitywerekept.MoreinformationaboutusingICAtoreducethe informativeforouranalysis,wewouldnotobtainanideaofthe
contributionofartifactcanbefoundinJungetal.(2000). locusofactivationorthepatternofactivationoverthescalp.
Event-relatedpotential(ERP)componentshavebeenshownto We focused our analysis on the windows 150–275 ms post
indexattention(Callaway&Halliday,1982;Harter&Aine,1984; stimulus-onsetinthecueandresponseintervals.Thesewindows
Lucketal.,2000),inparticulartheP200andN200latenciesand werefoundtocontainP200andN200ERPcomponents.Oneach
|     |     |     |     |     |     |     | trial, we | measured | the | peak | positive | and | negative | amplitudes, |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | -------- | --- | ---- | -------- | --- | -------- | ----------- | --- |
amplitudes,andthesevalueswereusedasindependentmeasures
|              |                  |     |           |               |     |            | and the | latency | at which | these | peaks | were | observed. | We  | used |
| ------------ | ---------------- | --- | --------- | ------------- | --- | ---------- | ------- | ------- | -------- | ----- | ----- | ---- | --------- | --- | ---- |
| of attention | in the following |     | analyses. | Event-related |     | potentials |         |         |          |       |       |      |           |     |      |
these8single-trialmeasurestopredictsingle-trialdiffusionmodel
| (ERPs) are | EEG responses | that | are time-locked |     | to  | a stimulus |     |     |     |     |     |     |     |     |     |
| ---------- | ------------- | ---- | --------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parameters.Howeverinthispaperwewillfocusonlyontheresults
onsetandaretypicallyestimatedbyaligningandaveragingEEG
ofmodelswith4single-trialmeasures:theamplitudeandlatency
responsesacrosstrials.Theyusuallycannotbedirectlymeasured
on each trial from single electrodes. Raw EEG signals could be of the peak positive deflection (P200) during the cue interval
usedasasingle-trialmeasuresbuttypicallyhaveverylowsignal- and the amplitude and latency of the peak negative deflection
to-noise ratios (SNR) for task-specific brain responses. Since the (N200)duringtheresponseinterval,becauseveryweakevidence,
goalofthisanalysiswastoexploresingle-trialeffectsofattention ifany,wasfoundfortheeffectsoftheotherattentionmeasures
onvisualdecisionmaking,asingle-trialestimateoftheERPwas on diffusion model parameters in models with all 8 single-trial
developed. measures. It should be noted that single-trial measures of EEG
Because the signal-to-noise ratio (SNR) in ongoing EEG in- spectral responses at SSVEP frequencies (see Nunez et al., 2015)
|     |     |     |     |     |     |     | were briefly | explored | but | future | methods | must | be  | developed | to  |
| --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --- | ------ | ------- | ---- | --- | --------- | --- |
creaseswhenadjacentelectrodesofrelevantactivityaresummed
increasesignal-to-noiseratiosofSSVEPmeasuresonsingle-trials.
(Parra,Spence,Gerson,&Sajda,2005),weanticipatedthattheSNR
oftheresponsetothevisualstimuluswouldbeboostedonindi-
vidualtrialsbysummingoverthemixtureofchannelsthatbest 2.3. HierarchicalBayesianmodels
| described | the average visual | response. |     | Traditional | ERPs | at each |     |     |     |     |     |     |     |     |     |
| --------- | ------------------ | --------- | --- | ----------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
channel(representedbyamatrixofsizeT×CwhereTisthelength Hierarchical models of visual decision making were assumed
ofatrialinmillisecondsandCisthenumberofEEGchannels)were and placed into a Bayesian framework. Bayesian methods yield
calculatedseparatelyforeachsubject.OneERPwascalculatedfor a number of benefits compared to other inferential techniques
theresponsetothevisualsignalandanotherwasfoundforthere- such as traditional maximum likelihood methods. Rather than
sponsetothevisualnoisebyaveragingarandomsetoftwo-thirds point estimates of parameters, Bayesian methods provide entire

122 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
Fig.3. Avisualrepresentationofthesingularvaluedecomposition(SVD)methodforfindingsingle-trialestimatesofevokedresponsesinEEG.TheEEGpresentedhereis
time-lockedtothesignalonsetduringtheresponseinterval,suchthatthesingle-trialERPencodedtheresponsetothesignalonset.AsingletrialofEEGfromSubject16
(Left)canbethoughtofasatimebychannel(T×C)matrix.ThefirstSVDcomponentexplainedthemostvariability(79.9%)inSubject16’sERPresponsetothesignalacross
alltrialsinthetrainingset.SVDweightsv(C×1)areobtainedfromtheERPresponse(i.e.trial-averagedEEG;T×C)andcanbeplottedonacartoonrepresentationofthe
humanscalpwithintermediateinterpolatedvalues(Middle).Thisspecifictrial’sERP(Right)wasobtainedbymultiplyingthetimeseriesdatafromeachchannelonthistrial
bytheassociatedweightinvectorvandthensummingacrossallweightedchannels.
distributionsoftheunknownparameters.Bayesianmethodsalso oneelementperEEGregressorandeacheffectofEEGisassumed
allowustoperformthemodelfittingprocedureinasinglestep, tobestatisticallyindependentfromtheothers.Ifadrift-diffusion
maintaining all uncertainty about each parameter through each modelparameterwasassumedtobeequaltoalinearcombination
hierarchicallevelofthemodel. ofEEGinputsthenthefollowingtwolinesreplacedthepriorsof
One downsideof Bayesianmethods isthat creatingsampling therespectiveparameterabove.
| a lg o r i t hm | s to fin | d p o st e | r io r d is tri | b u ti o n | so f B a y | es i an h | ie ra r c h i c a l |               |           |              |                    |              |     |
| --------------- | -------- | ---------- | --------------- | ---------- | ---------- | --------- | ------------------- | ------------- | --------- | ------------ | ------------------ | ------------ | --- |
|                 |          |            |                 |            |            |           |                     | η |µ (η)k ,σ | (η) ∼N(µ | (η)k ,σ 2 ), | µ (η)k ∼N(0,1002), | σ (η)∼Γ(5,5) |     |
m o d e l s ca n be ti m e c o n s u m in g a n d c u m b er s o m e . Ho w e v e r J u s t jk ( η)
|     |     |     |     |     |     |     |     | γ  | ∼MVN(µ |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
AnotherGibbsSampler(JAGS;Plummeretal.,2003)isaprogram |µ (γ)k ,σ (γ) (γ)k ,σ ( 2 γ)I), µ (γ)k ∼MVN(0,1002I), σ (γ)∼Γ(5,5).
jk
thatusesmultiplesamplingtechniquestofindestimatesofhier-
InModels2and3,theparameteroneachtrialwasassumedto
archicalmodels,onlyrequiringtheformofthemodel,data,and
beequaltoasimplelinearcombinationofthevectorofsingle-trial
initialvaluesasinputfromtheuser.Inordertofindposteriordis- η γ
|     |     |     |     |     |     |     |     | EEGinputsx | onthattrialiwith |     | and | astheinterceptand |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---------------- | --- | --- | ----------------- | --- |
|     |     |     |     |     |     |     |     |            | ijk              |     | jk  | jk                |     |
tributions,wehaveusedJAGSwithanextensionthataddsadiffu- slopesrespectively:
sionmodeldistribution(withoutintrinsictrial-to-trialvariability)
|     |     |     |     |     |     |     |     | δ =η | +γ(cid:124) | x   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- | --- | --- | --- |
asoneofthedistributionstobesampledfrom(Wabersich&Van- ijk (δ)jk (δ)jk ijk
d e k e r c k ho v e , 2 0 1 4 ) . S im il ar s o ft w a r e p a c k a g e s t o f it hi e r a rc h i c a l τ =η +γ(cid:124) x
|                 |             |           |             |              |             |             |                   | ijk (τ)jk | (τ)jk       | ijk |     |     |     |
| --------------- | ----------- | --------- | ----------- | ------------ | ----------- | ----------- | ----------------- | --------- | ----------- | --- | --- | --- | --- |
| d if fu s i o n | m o d e l s | h a v e b | ee n de v e | l op e d i n | d e p e n d | e n tl y in | o t h e r p r o - |           |             |     |     |     |     |
|                 |             |           |             |              |             |             |                   | ς =η      | +γ(cid:124) | x   |     |     |     |
gramminglanguagessuchasPython(Wiecki,Sofer,&Frank,2013). ijk (ς)jk (ς)jk ijk
| In order | to evaluate |     | the benefit | to prediction |     | of adding | EEG |     |     |     |     |     |     |
| -------- | ----------- | --- | ----------- | ------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
wherethefirsttwoequationsrefertothestructureofModel2and
measurestohierarchicaldiffusionmodels,threedifferentmodels
allthreeequationsrefertothestructureofModel3.Notethatthe
| were compared. |     | Model | 3 assumed | that | evidence | accumulation |     |                    |     | γ                                |     |     |     |
| -------------- | --- | ----- | --------- | ---- | -------- | ------------ | --- | ------------------ | --- | -------------------------------- | --- | --- | --- |
|                |     |       |           |      |          |              |     | p∗1vectorofeffects |     | ofEEGoneachparametercouldinclude |     |     |     |
rates, evidence accumulation variances, and non-decision times jk
|     |     |     |     |     |     |     |     | theinterceptterm |     | η to createa | ( p+1 )∗1vectorofeffects |     | γ∗  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | ------------------------ | --- | --- |
|     |     |     |     |     |     |     |     |                  |     | jk           |                          |     | jk  |
were each equal to a linear combination of EEG measures on (andtheEEGvectorx wouldincludeavalueof1tobemultiplied
ijk
eachtrial.Becausewefoundnoeffectoftheobservedsingle-trial bytheinterceptterm).WeusethisnotationinFig.4forsimplicity.
EEG measures on single-trial evidence accumulation variances, Becausenotalltrialsarebelievedtoactuallycontainadecision-
| we also | fit Model | 2, where | single-trial |     | evidence | accumulation |     |     |     |     |     |     |     |
| ------- | --------- | -------- | ------------ | --- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
makingprocess(i.e.thesubjectquicklypressesarandombutton
ratesandnon-decisiontimeswereinfluencebyEEG,butsingle-
|                  |           |        |             |       |       |            |      | during a | certain     | percentage | of trials reflecting | a ‘‘fast     | guess’’), |
| ---------------- | --------- | ------ | ----------- | ----- | ----- | ---------- | ---- | -------- | ----------- | ---------- | -------------------- | ------------ | --------- |
| trial evidence   | variances |        | were not.   | Model | 1 did | not assume | any  |          |             |            |                      |              |           |
|                  |           |        |             |       |       |            |      | reaction | times below | a certain  | threshold            | were removed | from      |
| EEG contribution |           | to any | parameters. | This  | model | assumed    | that |          |             |            |                      |              |           |
analysisandcross-validation.Cutoffreactiontimeswerefoundfor
parametersnotvaryingwithEEGwouldchangebasedonsubject
eachsubjectbyusinganexponentialmovingaverageofaccuracy
andcondition,drawnfromaconditionleveldistribution.Graphical
|     |     |     |     |     |     |     |     | after sorting | by  | reaction | time (Vandekerckhove | & Tuerlinckx, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | -------------------- | ------------- | --- |
representationsofthehierarchicalBayesianmodelsareprovided
2007).Therejectedreactiontimeswereallbelow511mswitha
inFig.4followingtheconventionofLeeandWagenmakers(2014).
meancutoffof410msacrosssubjects.Thisresultedinanaverage
For Model 1 (Fig. 4(a)), prior distributions were kept mostly rejectionrateof1.4%oftrialsacrosssubjectswithamaximumof
uninformative(i.e.parametersofinteresthadpriordistributions 6.3%oftrialsrejectedforonesubjectandaminimumof0.7%of
withlargevariances)sothattheanalyseswouldbedata-driven. trialsrejectedfor11ofthe17subjects.
| The prior | distributions |          | of parameters |     | for each      | subject | j and     |              |        |           |                     |              |        |
| --------- | ------------- | -------- | ------------- | --- | ------------- | ------- | --------- | ------------ | ------ | --------- | ------------------- | ------------ | ------ |
|           |               |          |               |     |               |         |           | Each model   | was    | fit using | JAGS with six       | Markov Chain | Monte  |
| condition | k free        | from EEG | influence     | had | the following |         | prior and |              |        |           |                     |              |        |
|           |               |          |               |     |               |         |           | Carlo (MCMC) | chains | run       | in parallel (Tange, | 2011) of     | 52,000 |
hyperpriorstructure
sampleseachwith2000burn-insamplesandathinningparameter
δ |µ ,σ ∼N(µ ,σ 2 ), µ ∼N(1.5,42)∈(−9,9), σ (δ)∼Γ(5,0.20) of 10 resulting in 5000 posterior samples in each chain. The
| jk (δ)k | (δ) | (δ)k ( | δ) (δ)k |     |     |     |     |     |     |     |     |     |     |
| ------- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
τ ∼N(µ p o s t e r i o r s a m p l e s f r om e a c h c h a in w er e co m b ine d t o fo r m o n e
| |µ (τ)k ,σ | (τ) | (τ)k ,σ | 2 ), µ (τ)k | ∼N(0.3,12)∈(0,3), |     | σ (τ)∼Γ(5,0.05) |     |     |     |     |     |     |     |
| ---------- | --- | ------- | ----------- | ----------------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- |
jk ( τ) p o s t e r i o r s a m p l e o f 30 ,0 0 0 s a m pl es f or e a ch p ar a m et e r. A l l
| ς |µ ,σ | ∼N(µ | ,σ  | ), µ | ∼N(0.6,22)∈(0,4), |     | σ (ς)∼Γ(5,0.05) |     |     |     |     |     |     |     |
| -------- | ----- | --- | ---- | ----------------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- |
jk (ς)k (ς) (ς)k ( 2 ς) (ς)k three models converged as judged by R ˆ being less than 1.02
ˆ
wherethenormaldistributionsN areparameterizedwithmean for all parameters in each model. R is a statistic used to assess
Γ
and variance respectively and the gamma distributions are convergenceofMCMCalgorithms(Gelman&Rubin,1992).
parameterizedwithshapeandscaleparametersrespectively. Posterior distributions were found for each free parameter
InModels2(Fig.4(b))and3(Fig.4(c)),toensurenoninterfer- in the three models. Credible intervals of the found posterior
encebythepriordistributions,uninformativepriorsweregivenfor distributions were then calculated to summarize the findings of
boththeeffects γ ofEEGontheparametersofinterestandthe each model. EEG regressor effects were deemed to have weak
jk
linearintercepts η .NotethattheeffectofEEG γ isavectorwith evidence if the 95% credible interval between the 2.5th and
|     |     | jk  |     |     |     | jk  |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 123
(a)Model1.
|     | (b)Model2. |     |     |     | (c)Model3. |     |     |     |     |
| --- | ---------- | --- | --- | --- | ---------- | --- | --- | --- | --- |
Fig.4. GraphicalrepresentationsofthethreehierarchicalBayesianmodelsfollowingtheconventionofLeeandWagenmakers(2014).Eachnoderepresentsavariablein
themodelwitharrowsindicatingwhatvariablesareinfluencedbyothervariables.Themagenta2∗1vectorofreactiontimeandaccuracyyijkandtheblue(p+1)∗1vector
ofpEEGregressors(+1intercept)xijkareobservedvariables,asindicatedbytheshadednodes.Boldedbluevariablesindicate(p+1)∗1vectors,suchasthesubjectjlevel
| γ ∗ |     |     | µ   |     |     |     |     | τ   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
e ff e c ts o f ea c h E E G r e g re s s o r a n d th e c o n d it io n k l e v e le f f e ct s ( γ ∗ )k o f e a c h E E G r eg r e s s o r . In M o d e l 3 fo r e a ch t r i al i, v al u e s o f n o n - d e c i s io n t i m e i jk , d r i f t r at e ( e vi d e n c e
| jk  | δ   |     |     | ς   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ac c u m u la ti o n ra t e ) i jk, t h e d i f fu s io n c o ef fi c ie n t (e v id e n c e a c c u m u l a t io n v a r ia n c e ) i jk a r e d e t e r m in is t ic l in e ar c o m b i n at io n s o f s in g l e -t r i a l E E G r e g re s s o r s x i j k a n d t h e e ff e c t s
ofthoseregressorsγ∗ thatvarybysubjectandcondition.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthis
jk
article.)
µ
97.5thpercentilesofthesubjectmeanparameter (γ)j wasnon- 4/17 subjects as the test set. After fitting the model with the
overlappingzero.Effectsweredeemedtohavestrongevidenceif trainingset,posteriorpredictivedistributionsoftheaccuracy-RT
the99%credibleintervalbetweenthe0.5thand99.5thpercentiles datawerefoundforeachsubject.Posteriorpredictivedistributions
wasnon-overlappingzero. werecalculatedbydrawingfromthesubject-levelposteriorsofthe
knownsubjectsandbydrawingfromthecondition-levelposteriors
|     |     |     |     |     | of the unknown | subjects. | The posterior | predictive | distributions |
| --- | --- | --- | --- | --- | -------------- | --------- | ------------- | ---------- | ------------- |
2.4. Cross-validation
werethencomparedtothesampledistributionofthetestset.
Insomerecentpapers,evaluationofmodels’predictionability
Alltrialsfromallsubjectswereusedduringinitialexploration
|              |                 |             |                 |       | has been | left to the readers | with the   | aid of posterior | predictive |
| ------------ | --------------- | ----------- | --------------- | ----- | -------- | ------------------- | ---------- | ---------------- | ---------- |
| of the data. | However once it | was decided | that the signal | onset |          |                     |            |                  |            |
|              |                 |             |                 |       | coverage | plots (e.g. see     | figures in | Supplementary    | materials, |
responsewasacandidatepredictorofdriftrate,cross-validation
|               |                  |          |                |         | Appendix | A). Here we | formally evaluate | the | similarity of the |
| ------------- | ---------------- | -------- | -------------- | ------- | -------- | ----------- | ----------------- | --- | ----------------- |
| was performed | using a training | and test | set of trials. | Out-of- |          |             |                   |     |                   |
sampleperformanceforbothknownandunknownsubjectswere posterior predictive distributions to the test samples via a
found by randomly assigning two-thirds of the trials from each ‘‘proportion of variance explained’’ calculation. Specifically, we
calculatedR2
subject in a random sample of subjects (i.e. 13 of 17 subjects ofsubjects’accuracyandcorrectreactiontime25th
pred
werechosenatrandom)asthetrainingsetandone-thirdofthe percentiles,medians,and75thpercentilesacrosssubjects.R2 is
pred
trials from the 13/17 subjects and all trials from the remaining a measure of percentage variance in a statistic T (e.g. accuracy,

124 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
Fig.5. Single-trialevokedresponsesofanexamplesubject,Subject12,tothevisualnoiseduringthecueinterval(Left)andsingle-trialevokedresponsestothevisual
signalduringtheresponseinterval(Right).Single-trialP200andN200magnitudeswerefoundbyfindingpeakamplitudesin150–275mstimewindows(asindicatedby
theverticaldashedlines)oftheSVD-biasedEEGdatainboththecueandresponseintervals.Thefirst300msoftheintervalsaresortedbysingle-trialP200magnitudesin
thecueintervalandsingle-trialN200magnitudesintheresponseinterval.Latenciesofthesingle-trialP200andN200componentscorrespondtoknownlatenciesofP2and
N1ERPcomponents.
correct-RTmedian,etc.)explainedbyin-sampleorout-of-sample 3.1. Intercepttermsofevidenceaccumulationrateandnon-decision
| prediction.Inthispaper,R2 |     |     |     | isdefinedasthepercentageoftotal |     |     |     | time |     |     |     |     |
| ------------------------- | --- | --- | --- | ------------------------------- | --- | --- | --- | ---- | --- | --- | --- | --- |
pred
between-subjectvarianceofastatisticTexplainedbyout-of-sample
orin-sampleprediction.Itisafunctionofthemeansquarederror Theintercepttermofeachvariablegivesthevalueofeachvari-
of prediction (MSEP) and the sample variance of the statistic T ablenotexplainedbyalinearrelationshiptoN200andP200am-
= = plitudesandlatencies.Thatis,theinterceptgivesthevalueofeach
| basedon | asample | sizeofJ |     | 13orJ | 4subjects | forknown |     |                |         |          |            |                    |
| ------- | ------- | ------- | --- | ----- | --------- | -------- | --- | -------------- | ------- | -------- | ---------- | ------------------ |
|         |         |         |     |       |           |          |     | parameter that | remains | constant | from trial | to trial, with the |
andunknownsubjectcalculationsrespectively.Thismeasurealso
allows comparisons across studies with similar prediction goals. between-trialvariabilityoftheparameterbeinginfluencedbythe
Mathematically,R2 isdefinedas changingtrial-to-trialEEGmeasures.Model2’sposteriormedians
|     |     | pred |     |     |     |     |     |     |     |     |     | µ   |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
oftheconditionlevelevidenceaccumulationrateintercepts (ηδ)j
µ
 J a n d n o n - d e c i s i on t im e i n te rc e p t s ( η τ )j a r e re p o r t ed . In lo w n o is e
|     | ( T | −T(pred)j | )2/( | J−1 ) |     |     |     |     |     |     |     |     |
| --- | --- | --------- | ---- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
j co n d it i o n s , e v i d en c e ac c u m u la t i on r a t e i n te rc e p t s w e re 1 .4 6 e v -
|     | j=1 |     |     |     | MSEP |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
R2 =1− =1− T (2) idence units per second (i.e. if there was no behavioral effect of
| pred |     |     |     |     |    |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 J ¯)2/( V ar[T] EEG on each trial and no variance in the evidence accumulation
|     |     | ( T −T | J−1 | )   |     |     |     |                                                       |     |     |     |                   |
| --- | --- | ------ | --- | --- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | ----------------- |
|     |     | j      |     |     |     |     |     | process,itwouldtaketheaveragesubject343mstoaccumulate |     |     |     |                   |
|     | j=1 |        |     |     |     |     |     |                                                       |     |     | α   |                   |
|     |     |        |     |     |     |     |     | evidencesinceadecisionisreachedwhen                   |     |     |     | = 1evidenceunitis |
accumulatedandsubjectsstarttheevidenceaccumulationprocess
with0.5evidenceunits).Inmediumandhighnoiseconditions,ev-
3. Results
idenceaccumulationrateinterceptswere1.30and0.86evidence
unitspersecondrespectively.Non-decisiontimeinterceptswere
Thesingle-trialEEGmeasures‘‘regressed’’ondiffusionmodel
340msinlownoiseconditions,425msinmediumnoisecondi-
parameterswerethepeakpositiveandnegativeamplitudesand
tions,and440msinhighnoiseconditions.
latencies(correspondingtoP200andN200peaksrespectively)in
|     |     |     |     |     |     |     |     | To understand | the | degree | of influence | of EEG on model |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------ | ------------ | --------------- |
the150–275mswindowspostnoise-onsetinthecueintervaland
|     |     |     |     |     |     |     |     | parameters, | approximate | condition | level evidence | accumulation |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | --------- | -------------- | ------------ |
postsignal-onsetintheresponseinterval.Howeverthemagnitude
ratesandnon-decisiontimeswerecalculatedandthencompared
| and latency | of the | peak | negative | deflection | (N200) | in  | response |                  |        |            |            |                 |
| ----------- | ------ | ---- | -------- | ---------- | ------ | --- | -------- | ---------------- | ------ | ---------- | ---------- | --------------- |
|             |        |      |          |            |        |     |          | to the intercept | of the | respective | parameter. | Taking the mean |
tothenoisestimulusandthemagnitudeandlatencyofthepeak
peakpositiveandpeaknegativeamplitudesandlatenciesacross
positivedeflection(P200)inresponsetothesignalstimuluswere
allsubjectsandtrialsineachnoiseconditionandmultiplyingby
notinformative(i.e.mostcondition-leveleffectposteriorsofthese
|          |            |      |               |     |           |      |          | the median | posterior | of the effects, | it was | found that evidence |
| -------- | ---------- | ---- | ------------- | --- | --------- | ---- | -------- | ---------- | --------- | --------------- | ------ | ------------------- |
| measures | overlapped | zero | significantly |     | in models | with | all P200 |            |           |                 |        |                     |
accumulaterateinlownoisewas1.90evidenceunitspersecond,
| and N200 | measures | included |     | as regressors). |     | For simplicity | we  |     |     |     |     |     |
| -------- | -------- | -------- | --- | --------------- | --- | -------------- | --- | --- | --- | --- | --- | --- |
1.65evidenceunitspersecondinmediumnoise,and1.35evidence
onlydiscussresultsofmodelswithP200measuresfollowingthe
unitspersecondinhighnoise.Itwasalsofoundthatnon-decision
noise stimulus in the cue interval and N200 measures following timewas393,400,and425msinthelow,medium,andhighnoise
thesignalstimulusintheresponseinterval.Examplesingletrial conditions. The intercepts of non-decision time thus described
amplitudesoftheseP200andN200peaksforSubject12areshown approximately86%,94%,and96%ofthetrueconditionmeansin
inFig.5. low, medium, high noise conditions respectively. However, the
Sincenoeffectsofexploredmeasureswerefoundonwithin- intercepts of the drift rates only described approximately 77%,
trial evidence accumulation variance in Model 3 (i.e. posterior 79%, and 63% of the true condition means in low, medium, and
| distributions | of  | γ   | were centered |     | near zero), | the only | EEG |     |     |     |     |     |
| ------------- | --- | --- | ------------- | --- | ----------- | -------- | --- | --- | --- | --- | --- | --- |
(ς)jk highnoiseconditionsrespectively.Whilethisgivesthereaderan
effectsthatwillbediscussedarethoseonevidenceaccumulation
ideaofthestrengthoftheinfluenceofsingle-trialEEGmeasures
| rate and | non-decision |     | time from | a fit | of Model | 2. The | posterior |     |     |     |     |     |
| -------- | ------------ | --- | --------- | ----- | -------- | ------ | --------- | --- | --- | --- | --- | --- |
ontheparameters,betterevaluationsofthedegreeofeffectsare
| distributions | of EEG | effects | on  | evidence | accumulation | rate | γ     |                 |     |     |     |     |
| ------------- | ------ | ------- | --- | -------- | ------------ | ---- | ----- | --------------- | --- | --- | --- | --- |
|               |        |         |     |          |              |      | (δ)jk | presentedbelow. |     |     |     |     |
γ
| andnon-decisiontime |     |     | (τ)jk didnotdiffersignificantlyfromModel |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2toModel3.Moreover,Model2producedbetterout-of-sample 3.2. Effectsofattentiononnon-decisiontimeinlow-noiseconditions
| prediction | than | Model | 3 for | new subjects. | A   | graphical | example |     |     |     |     |     |
| ---------- | ---- | ----- | ----- | ------------- | --- | --------- | ------- | --- | --- | --- | --- | --- |
oftheeffectsfoundwithModel2intworepresentativetrialsare Strong evidence was found to suggest that in low noise
giveninFig.1. conditions single-trial non-decision times τ are positively
ijk

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 125
Fig.6. Theposteriordistributionsoftheeffectofatrial’sN200latencyduringtheresponseinterval(onsetattentionlatencytothesignalstimulus)ontrial-specificnon-
decisiontimesτ ijkforeachsubjectjinthelownoiseconditionk = 1.Subjects2,6,7and11wereleftoutofthetrainingsetandtheirpredictedposteriordistributions
areshowninred.Thicklinesformingthedistributionfunctionsrepresent95%credibleintervalswhilethinlinesrepresent99%credibleintervals.Crossesandverticallines
representposteriormeansandmodesrespectively.AlsoshownarethetopographicrepresentationsofthechannelweightsofthefirstSVDcomponentofeachsubject’s
responseintervalERP,indicatingthelocationofsingle-trialN200soveroccipitalandparietalelectrodes.Evidencesuggeststhatlongerattentionallatenciestothesignal,
N200latencies,arelinearlycorrelatedwithlongernon-decisiontimesinthelownoisecondition.(Forinterpretationofthereferencestocolorinthisfigurelegend,the
readerisreferredtothewebversionofthisarticle.)
linearlyrelatedtodelaysintheEEGresponsetothevisualsignal driftrate,wasfoundtohavea95%credibleintervalof0.02–0.34
asmeasuredbythelatencyofthenegativepeak(N200)following evidenceunitspersecondincrease(whereittakes α =1evidence
stimulus onset. A probability greater than 99% of the condition- unittomakeadecision)andaposteriormedianof0.17evidence
level effect being greater than zero in all subjects was found. unitspersecondincreaseforeachmagnitudeincrease(i.e.away
. µ
ThisrelationshiptoanEEGsignature150–275mspoststimulus from zero) of 2683 V, the standard deviation of the negative
onsetsuggestsaneffectonpreprocessingtimeratherthanmotor- peak. Given the same magnitude increase, the posterior median
responsetime.Byexploringtheposteriordistributionofthemean oftheeffectsinthemediumandhighnoiseconditionswere0.13
µ
effect across-participants (γτ)j , it is inferred that non-decision and0.14evidenceunitspersecondrespectivelywith95%credible
timeincreases12ms(theposteriormedian)whenthereisa40ms intervals−0.01to0.28and0–0.28respectively.
increase in the latency of the single-trial negative peak (where Strong evidence was found to suggest that the magnitude of
40mswasthestandarddeviationacrossalltrialsandsubjects)in the positive peak of the response to the visual noise during the
thelownoisecondition,witha99%credibleintervalof3–21ms. cueintervalaffectedthefutureevidenceaccumulationrateinthe
Fig. 6 shows the per-subject effects of signal N200 latency on mediumnoiseandpossiblyhighnoiseconditions.Themedianof
non-decision time in the low noise condition. No evidence was the posterior distribution of the condition-level effect was 0.20
foundtosuggestthatthesignalN200latencyaffectednon-decision evidenceunitspersecondwhentherewasa27 . 67 µ Vincrease,the
timeinmediumnorhighnoiseconditions.95%credibleintervals standarddeviationofthepeakmagnitude.A99%credibleinterval
for these increases in the subject mean non-decision time for ofthiseffectwas0.04–0.32evidenceunitspersecond.Fig.7shows
40 ms increased N200 delays were −9 to 4 ms and −8 to 3 ms theeffectsofthenoiseP200amplitudesonspecificsubjects’single-
respectively. No evidence was found to suggest that attentional trial drift rates in the medium noise condition. The probability
delaytothenoise,thenoiseP200latency,affectednon-decision of there being an effect of this P200 amplitude during the cue
time. intervalinthehighnoiseconditionwas94.6%(i.e.theamountof
Weak evidence was found to suggest that magnitude of the theposteriordensityofthecondition-leveleffectabovezero).The
responsetothestimulusaffectsnon-decisiontimeinthelownoise medianoftheposteriordistributionofthiseffectwas0.09evidence
condition. The posterior median suggests that a 26 . 83 µ V (the units per second with a 95% credible interval of −0.02 to 0.22
. µ
standard deviation) decrease in magnitude of the negative peak evidence units per second when there is a 2767 V increase in
(i.e.movesthenegativepeaktowardzero)leadstoa11msincrease ahighnoisetrial.
in non-decision time. The 95% credible interval of this effect of
N200signalmagnitudeonnon-decisiontimewasa2–21ms.No 3.4. Cross-validation
evidencewasfoundtosuggestthatthemagnitudeinthemedium
andhighnoiseconditionsaffectednon-decisiontime. In-sample and out-of-sample posterior predictive coverage
plots of correct-RT distributions for each condition and subject
3.3. Effectsofattentiononevidenceaccumulation areprovidedintheSupplementarymaterials(seeAppendixA).All
threemodelsperformwellatpredictingcorrect-RTdistributions
Evidence was found to suggest that per-trial response to the andoverallaccuracyoftrainingdata(i.e.in-sampleprediction;see
visual signal (measured by the negative peak, N200, amplitude) Table3andDiscussionsection).Howeverbycross-validationwe
ispositivelycorrelatedwithper-trialevidenceaccumulationrates foundthattheadditionofsingle-trialEEGmeasuresofattentional
δ µ
ijk in each condition. In the low noise condition, (γ(δ))j , which onsetimprovedout-of-samplepredictionofaccuracyandcorrect
describestheacross-subjectmeanoftheeffectofnegativepeakon reactiontimedistributionsofknownsubjects(i.e.thosesubjects

126 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
Fig.7. Theposteriordistributionsoftheeffectofatrial’sP200amplitudeduringthecueinterval(onsetofattentiontothenoisestimulus)ontrial-specificevidence
accumulationratesδ ijkforeachsubjectjinthemediumnoiseconditionk=2.Subjects2,6,7and11wereleftoutofthetrainingset,theirpredictedposteriordistributions
areshowninred.Thicklinesformingthedistributionfunctionsrepresent95%credibleintervalswhilethinlinesrepresent99%credibleintervals.Crossesandverticallines
representposteriormeansandmodesrespectively.AlsoshownarethetopographicrepresentationofthechannelweightsofthefirstSVDcomponentofeachsubject’scue
intervalERP,indicatingthelocationofsingle-trialP200soveroccipitalandparietalelectrodes.Evidencesuggeststhattheeffectoftheattentiontothenoise,reflectedin
P200amplitudes,positivelyinfluencedthedriftrateofeachsubjectineachtrial,inthemediumandhighnoiseconditions.(Forinterpretationofthereferencestocolorin
thisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
| whohad2 | / 3oftheirtrialsusedinthetrainingset).R2 |     |     |     |     | indicates | Table1 |     |     |     |     |     |     |
| ------- | ---------------------------------------- | --- | --- | --- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- |
pred
the percentage of variance explained by prediction in the given Percentageofacross-subjectvarianceexplainedbyout-of-sampleprediction(R2 )
pred
statisticacrosssubjects.Table1containsR2 valuesforaccuracy foraccuracyandsummarystatisticsofcorrect-RTdistributionsofthosesubjects’
pred
aswellassummarystatisticsofcorrect-RTdistributionsofknown thatwereincludedinthetrainingset.13ofthesubjects’dataweresplitinto
2/3trainingand1/3testsets.Posteriorpredictivedistributionsthatpredictedtest
subjects.PredictionwasimprovedwhenusingModel3forthese set behavior were generated for 13 of the subjects by drawing from posterior
subjectswithatleast77.3%ofvarianceincorrect-RTmediansbeing distributionsgeneratedbythetrainingset.IntheLow,Medium,andHighnoise
conditions,the25th,50th(themedian),and75thpercentilesandmeanswere
| explained | by out-of-sample | prediction, |     | but Model | 2 performed |     |           |            |         |           |         |              |                  |
| --------- | ---------------- | ----------- | --- | --------- | ----------- | --- | --------- | ---------- | ------- | --------- | ------- | ------------ | ---------------- |
|           |                  |             |     |           |             |     | predicted | reasonably | well by | the model | without | single-trial | measures of EEG, |
almostaswellincomparisontoModel1,themodelwithoutsingle-
|           |               |            |     |            |       |          | Model 1.      | However including | single-trial |     | measures | of EEG improved  | prediction |
| --------- | ------------- | ---------- | --- | ---------- | ----- | -------- | ------------- | ----------------- | ------------ | --- | -------- | ---------------- | ---------- |
| trial EEG | inputs. Model | 2 was able | to  | predict at | least | 76.3% of |               |                   |              |     |          |                  |            |
|           |               |            |     |            |       |          | of correct-RT | distributions,    | especially   | in  | the Low  | noise condition, | with Model |
the variance in correct-RT medians while Model 1 was able to 3(whichassumesevidenceaccumulationrate,non-decisiontime,andevidence
predict at least 74.5% of the variance in correct-RT medians. In accumulationvariancevarywithEEGper-trial)onlyslightlyoutperformingModel
the low noise condition, Model 2 did not improve upon Model 2(whichassumesevidenceaccumulationrateandnon-decisiontimevarywithEEG
per-trial).
| 1’s explanation | of variance | in subject-level |     | accuracy | but | better |     |     |     |     |     |     |     |
| --------------- | ----------- | ---------------- | --- | -------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
Predictionofnewdatafromknownsubjects
predictedaccuracyintheothertwoconditions.
Larger gains in out-of-sample prediction were found for Model1 Model2 Model3
|         |                |                |     |      |          |        |     |     |     | Comparison |     | EEG-δ,τ | EEG-δ,τ,ς |
| ------- | -------------- | -------------- | --- | ---- | -------- | ------ | --- | --- | --- | ---------- | --- | ------- | --------- |
| unknown | subjects (i.e. | those subjects | who | were | not used | in the |     |     |     |            |     |         |           |
trainingset).Theseimprovementswereparticularlypronounced 25tht1percentile 81.6% 85.2% 85.6%
inthelownoisecondition.Model2outperformedModel3,which t1median 74.5% 76.3% 77.3%
|              |               |             |       |            |         |         | Low | 75tht1percentile |     | 60.2% |     | 63.3% | 63.8% |
| ------------ | ------------- | ----------- | ----- | ---------- | ------- | ------- | --- | ---------------- | --- | ----- | --- | ----- | ----- |
| outperformed | Model 1       | in turn, as | shown | in Table   | 2. From | these   |     |                  |     |       |     |       |       |
|              |               |             |       |            |         |         |     | Accuracy         |     | 24.5% |     | 20.8% | 27.7% |
| results it   | is clear that | Model 2 was | the   | best model | for     | out-of- |     |                  |     |       |     |       |       |
samplepredictionoverall,especiallyfornewsubjects.Inthelow 25tht1percentile 84.1% 85.1% 86.1%
|                  |       |            |            |       |             |     |        | t1median         |     | 86.8% |     | 87.6% | 88.9% |
| ---------------- | ----- | ---------- | ---------- | ----- | ----------- | --- | ------ | ---------------- | --- | ----- | --- | ----- | ----- |
| noise condition, | Model | 2 was able | to explain | 22.1% | of between- |     | Medium |                  |     |       |     |       |       |
|                  |       |            |            |       |             |     |        | 75tht1percentile |     | 63.1% |     | 68.2% | 69.3% |
subject variance in correct-RT 25th percentiles in the low-noise Accuracy 58.1% 63.5% 63.5%
| condition | while Model | 1 was not | able | to predict | any between- |     |     |         |                 |       |     |           |           |
| --------- | ----------- | --------- | ---- | ---------- | ------------ | --- | --- | ------- | --------------- | ----- | --- | --------- | --------- |
|           |             |           |      |            |              |     |     | 25tht 1 | p e r ce n tile | 7 3 . | 0 % | 7 6 . 4 % | 7 6 . 6 % |
subject variance in this measure. While the included single-trial t m e d ia n 7 7 . 4 % 7 6 . 8 % 7 7 . 8 %
|     |     |     |     |     |     |     | High | 1   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
EEG measures in these type of models do not perform as well 75tht1percentile 71.0% 74.2% 74.2%
as new subject prediction when subject-average EEG measures Accuracy 46.3% 48.9% 51.3%
| of attention                              | are included | (Nunez | et al., | 2015), | single-trial | EEG |                                                         |     |     |     |     |     |     |
| ----------------------------------------- | ------------ | ------ | ------- | ------ | ------------ | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| doesimproveprediction.TheimprovementsinR2 |              |        |         |        | acrossmodels |     |                                                         |     |     |     |     |     |     |
|                                           |              |        |         | pred   |              |     | amplitudes150–275msinthenoiseandresponseintervalrespec- |     |     |     |     |     |     |
suggestthatitispossiblethatsimilarmodelswithmoresingle-trial
|     |     |     |     |     |     |     | tively) | were averaged | across | trials | for each | subject. | Localization |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | ------ | ------ | -------- | -------- | ------------ |
measuresofEEGcouldexplainnewsubjects’accuracyandcorrect-
intimeandonthebrainarebasedontheseacross-trialaverages.
RTdistributions.
ThesubjectmeanandstandarddeviationoftheP200latencydur-
ingthecueintervalwas220±12mswhilethemeanandstan-
3.5. P200andN200localizations dard deviation the of N200 latency during the response interval
was217±6ms.Althoughtheselatenciesdifferslightlyfromtra-
Boththesubject-averageP200andN200componentswerelo- ditionalP2andN1findings(Lucketal.,2000),whenviewingthe
calized in time and space. All single-trial P200 and N200 ampli- event-relatedwaveformsoveralltrialsitisclearthattheP2andN1
tudes and latencies (i.e. single-trial peak positive and negative areinfluencedbythesingle-trialmeasures.Asanexample,every

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 127
Table2 withvisualattention(Buschman&Miller,2007;Corbetta&Shul-
Percentageofacross-subjectvarianceexplainedbyout-of-sampleprediction(R2 ) man,2002;Desimone&Duncan,1995)anddecisionmaking(Mul-
pred
foraccuracyandsummarystatisticsofnewsubjects’andcorrect-RTdistributions. der et al., 2014). Corroborating our findings, White et al. (2014)
Posteriorpredictivedistributionsweregeneratedfor4newsubjectsbydrawing
from condition level posterior distributions. Most R2 measures are negative found that blood-oxygen-level dependent (BOLD) activity in the
p re d
becausetheamountofvarianceinpredictionwasgr e a terthanthevarianceof right temporal superior gyrus, right angular gyrus, and areas in
themeasureacrosssubjects;howevertherelativevaluesfromonemodeltothe the right lateral occipital cortex (e.g. the right middle occipital
nextarestillinformativeabouttheimprovementinpredictionability.Themodel gyrus) correlated with non-decision time during a simple visual
withoutsingle-trialEEGmeasures,Model1,doesnotpredictnewsubjects’correct-
andauditorydecisionmakingtask.Itwashypothesizedthatthis
RTdistributions.Modelswithsingle-trialEEGmeasuresofonsetattention,Model2
activitywasduetomotorpreparationtimeinsteadofvisualpre-
andModel3,canpredictsomevarianceofthenewsubjects’25thpercentiles,with
Model2outperformingModel3. processing time (White et al., 2014); however the time-scale of
BOLDsignalsdoesnotprovideadditionalknowledgetoseparate
Predictionofnewdatafromnewsubjects
visualpreprocessingtimefrommotorpreparationtime.Informed
Model1 Model2 Model3 byEEG,BOLDsignalsassociatedwithevidenceaccumulationrates
|     |     |     | Comparison |     | EEG-δ,τ |     | EEG-δ,τ,ς |     |     |     |     |     |     |     |     |
| --- | --- | --- | ---------- | --- | ------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
havebeenpreviouslylocalizedtorightandleftsuperiortemporal
25tht1percentile −9.8% 22.1% 17.9% gyriandlateraloccipitalcorticalareas,thoughttocorrespondto
|     |     |              | −   |         | −   |         | −1      |     |     |     |     |     |     |     |     |
| --- | --- | ------------ | --- | ------- | --- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| Low |     | t 1 m e d ia | n 3 | 7 . 6 % | 1   | 1 . 6 % | 4 . 3 % |     |     |     |     |     |     |     |     |
75tht p e r ce n tile − 6 7 . 3 % − 4 6 . 7 % − 4 7 . 6 % earlybottom-upandlatetop-downdecisionmakingprocessesre-
1
Accuracy −8.1% −22.8% −65.9% spectivelyduringavisualface/cardiscriminationtask(Philiastides
&Sajda,2007).Therightandleftmiddleoccipitalgyrihavealso
|     | 25tht1percentile |            | −5.3% |         | 8.4% |         | 5.3%       |            |      |          |            |     |          |              |     |
| --- | ---------------- | ---------- | ----- | ------- | ---- | ------- | ---------- | ---------- | ---- | -------- | ---------- | --- | -------- | ------------ | --- |
|     |                  |            |       |         |      |         |            | previously | been | shown to | contribute | to  | evidence | accumulation |     |
|     |                  | t m e d ia | n − 3 | 0 . 7 % | − 2  | 2 . 5 % | −2 5 . 7 % |            |      |          |            |     |          |              |     |
Medium 1 − − − ratesduringarandomdotmotiontask(Turneretal.,2015).
|      | 75tht            | 1 p e r ce | n tile 6 | 2 . 6 % | 5      | 3 . 9 % | 5 6 . 2 % |               |     |     |     |     |     |     |     |
| ---- | ---------------- | ---------- | -------- | ------- | ------ | ------- | --------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
|      |                  | Accuracy   | −13.1%   |         | −35.5% |         | −67.7%    |               |     |     |     |     |     |     |     |
|      |                  |            | −1.7%    |         |        |         |           | 4. Discussion |     |     |     |     |     |     |     |
|      | 25tht1percentile |            |          |         | 8.2%   |         | 7.7%      |               |     |     |     |     |     |     |     |
|      |                  | t m e d ia | n − 1    | 2 . 0 % | − 4    | .3 %    | −2 . 9 %  |               |     |     |     |     |     |     |     |
| High |                  | 1          |          |         |        |         |           |               |     |     |     |     |     |     |     |
75tht p e r ce n tile − 3 7 . 6 % − 2 5 .5 % − 2 3 . 7% 4.1. Attentioninfluencesperceptualdecisionmakingoneachtrial
|     |     | 1   | −4.3% |     | −14.1% |     | −29.1% |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----- | --- | ------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Accuracy
Theresultsofthisstudysuggestthatfluctuationsinattention
toavisualsignalaccountsforsomeofthetrial-to-trialvariability
single-trialevokedresponseofSubject12tothenoiseandsignal
|     |     |     |     |     |     |     |     | in the brain’s | speed | of evidence |     | accumulation | on  | each | trial in |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ----- | ----------- | --- | ------------ | --- | ---- | -------- |
areshowninFig.5,sortedbypeakP200amplitudeinthecuein-
|     |     |     |     |     |     |     |     | each condition. | There | is  | also evidence |     | to suggest | that increased |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ----- | --- | ------------- | --- | ---------- | -------------- | --- |
tervalandsortedbypeakN200amplitudeintheresponseinterval.
responsetothecompetingvisualnoiseincreasesthebrain’sspeed
TheP200andN200latenciesofthissubjectcorrespondtotradi-
|     |     |     |     |     |     |     |     | of evidence | accumulation, |     | but only | in  | medium and | high | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------- | --- | -------- | --- | ---------- | ---- | ----- |
tionalP2andN1components.
|     |     |     |     |     |     |     |     | conditions. | Although | a simple | explanation |     | of this | effect | would |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | -------- | ----------- | --- | ------- | ------ | ----- |
WhileEEGlocalizationisaninexactprocessthatisunsolvable
|         |            |              |     |             |     |           |          | be differences | in  | trial-to-trial | arousal, |     | we note | that the | effect |
| ------- | ---------- | ------------ | --- | ----------- | --- | --------- | -------- | -------------- | --- | -------------- | -------- | --- | ------- | -------- | ------ |
| without | additional | assumptions, |     | the surface |     | Laplacian | has been |                |     |                |          |     |         |          |        |
onlyoccursinmedium-noiseandhigh-noiseconditions.Wehave
showntomatchcloselytosimulatedcorticalactivityusingforward
previouslyfoundevidencethatnoisesuppressionduringthecue
| models | (i.e. the | mapping | of cortical | activity | to  | scalp | potentials) |     |     |     |     |     |     |     |     |
| ------ | --------- | ------- | ----------- | -------- | --- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
intervalpredictsenhanceddriftratesbasedonthesubject-average
| and have | shown | consistent | results | when | used | with | real EEG |     |     |     |     |     |     |     |     |
| -------- | ----- | ---------- | ------- | ---- | ---- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
SSVEPresponsesinthesedata(Nunezetal.,2015).Thus,oureffect
data(Nunez&Srinivasan,2006).Unlike3Dsolutions,projections
|        |         |        |              |     |             |     |           | may reflect | the attention |     | the subject | places | on the | cue, | which |
| ------ | ------- | ------ | ------------ | --- | ----------- | --- | --------- | ----------- | ------------- | --- | ----------- | ------ | ------ | ---- | ----- |
| to the | surface | of the | cortex (more |     | accurately, | the | dura) are |             |               |     |             |        |        |      |       |
determineswhethertoengagemechanismsofnoisesuppression,
| theoretically | solvable, |     | and have | been | used with | success | in past |     |     |     |     |     |     |     |     |
| ------------- | --------- | --- | -------- | ---- | --------- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
butwecouldnotdirectlyassessthispossibilityasnewmethods
studies(seeNunezetal.,1994,foranexample).
mustbedevelopedtomeasureSSVEPsonsingletrials.Thiswould
Inthisstudywehavefoundsurfacespline-Laplacians(Nunez& reflectahypothesisbasedonthePerceptualTemplateModelthat
Pilgreen,1991)ontherealisticMNIaveragescalp(Deng,Winter, predictsthatsubjectswillsuppressattentiontovisualdistractors
Thorpe,&Srinivasan,2012)ofthemeanpositivepeakduringthe duringtasksofhighvisualnoise(Lu&Dosher,1998).
| cue interval | (the | P200) | and the | mean | negative | peak | during the |           |      |     |           |      |            |              |     |
| ------------ | ---- | ----- | ------- | ---- | -------- | ---- | ---------- | --------- | ---- | --- | --------- | ---- | ---------- | ------------ | --- |
|              |      |       |         |      |          |      |            | We assume | that | the | effect of | N200 | latency on | non-decision |     |
responseinterval(theN200)byaveragingovertrialsandsubjects.
timeduringtheresponseintervalisonpreprocessingtimeinstead
| The surface | Laplacians |       | were then | projected           |     | onto one | subject’s |          |          |       |           |     |                 |     |         |
| ----------- | ---------- | ----- | --------- | ------------------- | --- | -------- | --------- | -------- | -------- | ----- | --------- | --- | --------------- | --- | ------- |
|             |            |       |           |                     |     |          |           | of motor | response | time. | There was | no  | a priori reason | to  | believe |
| cortical    | surface    | using | Tikhonov  | (L2) regularization |     | and      | a Finite  |          |          |       |           |     |                 |     |         |
thatanattentioneffectthattakesplace150–275msafterstimulus
Element(FE;Pommier&Renard,2005)forwardmodeltotheMNI
onsetwouldaffectthespeedofefferentsignalstothemuscle,given
151averagehead,maintainingsimilardistributionsofactivityof
thattheresponsetimeswereatleast500ms.Thesefindingslead
thesurfaceLaplaciansonthecorticalsurface.Thesubject’sbrain ustoconcludethattheeffectofattentioninresponsetothesignal
was then labeled using the Destrieux cortical atlas (Fischl et al., in low noise conditions is to reduce preprocessing delay time.
2004).CorticaltopographicmapsofbothpeaksaregiveninFig.8. Thisappearsconsistentwithpredictionsofsignalenhancementin
BecauseweexpectthemajorityoftheLaplaciantooriginatefrom low-noiseconditionsinthePerceptualTemplateModel.However,
superficialgyri(Nunez&Srinivasan,2006),wehavelocalizedboth we note there is not a perfect equality between N200 latency
thepositiveandnegativepeaksonlytomaximallyactivegyri.This andpreprocessingtime(theconditionandsubject-levelcoefficient
| localization | suggested |     | that both | the P200 | and | N200 | were in the |            |         |          |        |     |                    |     |      |
| ------------ | --------- | --- | --------- | -------- | --- | ---- | ----------- | ---------- | ------- | -------- | ------ | --- | ------------------ | --- | ---- |
|              |           |     |           |          |     |      |             | posteriors | are not | centered | on 1). | And | the identification | of  | both |
followingextrastriateandparietalcorticallocations:rightandleft
preprocessingtimeandmotorresponsetimeisnotpossiblewith
middleoccipitalgyri,rightandleftsuperiorparietalgyri,rightand
|     |     |     |     |     |     |     |     | a drift-diffusion |     | model | fit without | additional | assumptions |     | or  |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ----- | ----------- | ---------- | ----------- | --- | --- |
leftangulargyri,theleftoccipitalinferiorgyrus,therightoccipital externalinputssuchasEEG.
superiorgyrus,andthelefttemporalsuperiorgyrus.Althoughwe Reactiontime(RT)andchoicebehaviorduringvisualdecision
should note that the exact localization must have some errors making tasks are well characterized by models that assume
due to between-subject variance in cortex and head shape and a continuous stochastic accumulation of evidence. And many
between-subjectvarianceintissueproperties. observations of increasing spike-rates of single neuron action
Brainregionsfoundusingthiscortical-Laplacianmethodpoint potentials lend support to this stochastic theory of evidence
toactivityinearlydorsalandventralpathwayregionsassociated accumulation on a neural level (Shadlen & Newsome, 1996,

128 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
Fig.8. Rightandleftsagittalandposteriorviewsoflocalizedsingle-trialP200evokedpotentialsduringthecueinterval(Top)andlocalizedsingle-trialN200evoked
potentialsduringtheresponseinterval(Bottom)averagedacrosstrialsandsubjects.ThecorticalmapswereobtainedbyprojectingMNI-scalpspline-Laplacians(Dengetal.,
2012;Nunez&Pilgreen,1991)ontoasubject’sanatomicalfMRIimageviaTikhonov(L2)regularization,maintainingsimilardistributionsofactivityofthesurfaceLaplacians
onthecorticalsurface.Blueandorangeregionsinmicroamperespermm2correspondtocorticalareasestimatedtoproducenegativeandpositivepotentialsobservedon
thescalprespectively.ThesetwoparticularprojectionsoftheLaplacianssuggestthatP200andN200activityoccursinextrastriatecorticesandareasintheparietallobe.
(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
2001). Recently some macroscopic recordings of the cortex withintheprobabilitydensityfunctionasassumedbyRatcliff(1978).
have shown that increasing EEG potentials ramping up to Instead we assume that each trial’s drift rate is exactly equal to
P300 amplitudes are correlates of the stochastic accumulation a linear function of EEG data and use an evidence accumulation
of evidence (O’Connell, Dockree, & Kelly, 2012; Philiastides, likelihood function that does not assume drift rates vary trial-
| Heekeren, | & Sajda,              | 2014; | Twomey, | Murphy,   | Kelly,   | & O’Connell, |     |           |              |        |      |           |              |            |           |
| --------- | --------------------- | ----- | ------- | --------- | -------- | ------------ | --- | --------- | ------------ | ------ | ---- | --------- | ------------ | ---------- | --------- |
|           |                       |       |         |           |          |              |     | to-trial  | by any other | means. |      | Per-trial | non-decision |            | times and |
| 2015). It | has been hypothesized |       |         | that this | EEG data | reflects     | the |           |              |        |      |           |              |            |           |
|           |                       |       |         |           |          |              |     | diffusion | coefficients | were   | also | assumed   | to           | be exactly | equal to  |
evidenceaccumulationprocessitself(oramixtureofthisprocess
linearfunctionsofEEGdata.
| with other | decision-making  |     | correlates) |      | and not    | a correlated |     |           |           |          |             |       |              |     |              |
| ---------- | ---------------- | --- | ----------- | ---- | ---------- | ------------ | --- | --------- | --------- | -------- | ----------- | ----- | ------------ | --- | ------------ |
|            |                  |     |             |      |            |              |     | Per trial | estimates | of       | diffusion   | model | parameters   |     | cannot be    |
| measure    | such as top-down |     | attention.  | This | hypothesis | leads        | to  |           |           |          |             |       |              |     |              |
|            |                  |     |             |      |            |              |     | obtained  | without   | imposing | constraints |       | or including |     | external in- |
the natural prediction that single-trial drift rates are explained puts.Inthisstudy,wehaveshownthatthesingle-trialP200and
by single-trial P300 slopes. However within a small region of N200 attention measures can be used to discover per trial esti-
the cortex, neurons will have diverse firing patterns during the mates of all three free parameters, non-decision time, drift rate,
decisionmakingprocess,onlysomeofwhichareobservedtohave andthediffusioncoefficient.Non-decisiontime,thedriftrate,and
increasingspiking-ratebehaviorindicativeofstochasticevidence theevidenceboundarycouldalsobemodeledaspertrialestimates
accumulation (Meister, Hennig, & Huk, 2013). The properties of of external inputs, as would be useful in other speeded reaction
volumeconductionthroughthecortex,skull,andskinonlyallow
|                 |               |       |               |     |                |           |        | time tasks     | where | external | per-trial | physiological |          | measurements |            |
| --------------- | ------------- | ----- | ------------- | --- | -------------- | --------- | ------ | -------------- | ----- | -------- | --------- | ------------- | -------- | ------------ | ---------- |
| for synchronous | post-synaptic |       | potentials    |     | to be observed |           | at the |                |       |          |           |               |          |              |            |
|                 |               |       |               |     |                |           |        | are available. | Other | possible | per-trial |               | external | inputs       | that could |
| scalp (Buzsaki, | 2006;         | Nunez | & Srinivasan, |     | 2006).         | Therefore | an     |                |       |          |           |               |          |              |            |
beusedinclude:magnetoencephalographic(MEG)measures,func-
increasingspike-rateasobservedonthesingle-neuronlevelisnot
|     |     |     |     |     |     |     |     | tional magnetic | resonance |     | imaging | (fMRI) | measures, |     | physiologi- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --------- | --- | ------- | ------ | --------- | --- | ----------- |
likelytobeobservedasanincreasingwaveforminEEGrecorded
calmeasurementssuchasgalvanicskinresponse(GSR),andnear
fromthescalp.Wealsowouldexpecttheevidenceaccumulation
|     |     |     |     |     |     |     |     | infrared | spectroscopy | (NIRS), |     | where | each modality |     | may have |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | ------- | --- | ----- | ------------- | --- | -------- |
processtoterminatebeforetheresponsetimesinceaportionofthe
|     |     |     |     |     |     |     |     | multiple | external | inputs | (e.g. | multiple | linear | EEG | regressors of |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------ | ----- | -------- | ------ | --- | ------------- |
responsetimemustbededicatedtothemotorresponseafterthe
single-trialparameters,asinthisstudy).Themoreexternalinputs
decisionismade.ThismaynotbethecasefortherampingP300
correlatedwithsingle-trialparametersincludedinthemodel,the
| waveform | on single-trials |     | even though |     | it is predictive | of  | model |     |     |     |     |     |     |     |     |
| -------- | ---------------- | --- | ----------- | --- | ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
betterthesingle-trialestimateoftheparameterswillbe.Thiswill
| parameters | (Philiastides | et  | al., 2014). | If  | the stochastic | evidence |     |     |     |     |     |     |     |     |     |
| ---------- | ------------- | --- | ----------- | --- | -------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
accumulationprocesswastrulyreflectedintherampingofEEG, allowdecisionmodelresearcherstobetterexploretheefficacyof
atestablepredictionwouldbethatthevariancearoundthemean thediffusionmodelbycomparingsingle-trialestimatesofparam-
| rateoftheP300ramponeachtrialwouldbelinearlyrelatedtothe |     |                                          |     |     |     |     |     | eters. |     |     |     |     |     |     |     |
| ------------------------------------------------------- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
| diffusioncoefficient                                    | ς   | ,inadditiontosingle-trialP300slopesbeing |     |     |     |     |     |        |     |     |     |     |     |     |     |
linearlyrelatedtothedriftrate δ .TherearemanyEEGmeasures 4.3. BehaviorpredictionandBCIapplications
thoughttoberelatedtoattentionsuchaseventrelatedpotential
|     |     |     |     |     |     |     |     | We have | observed | that | single-trial |     | measures |     | of EEG in a |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---- | ------------ | --- | -------- | --- | ----------- |
(ERP)components,powerincertainfrequencybands,andsteady-
|     |     |     |     |     |     |     |     | hierarchical | Bayesian | approach |     | to  | decision-making |     | modeling |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | -------- | --- | --- | --------------- | --- | -------- |
statevisualevokedpotential(SSVEP)responses.ItislikelythatEEG
improvesoverallaccuracyandcorrect-RTdistributionprediction
| measures | that share | similar | properties | with | stochastic | evidence |     |     |     |     |     |     |     |     |     |
| -------- | ---------- | ------- | ---------- | ---- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
accumulation processes are in fact due to these correlates of for subjects with observed behavior. This paradigm also lead to
attentionorotherformsofcorticalprocessingthatcaninfluence significantimprovementincorrect-RTdistributionpredictionfor
thedecisionmakingprocess. thosesubjectswhosebehaviorwasmissing.Thatis,wehaveshown
|     |     |     |     |     |     |     |     | that a new | subject’s | correct-RT |     | distributions |     | can | be predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | ---------- | --- | ------------- | --- | --- | ------------ |
4.2. External predictors allow for trial level estimation of diffusion when only their single-trial EEG is collected, given that other
modelparameters subjects’EEGandbehaviorhasbeenanalyzed.Ifthegoalofafuture
projectissolelyprediction(andnoexplanationofthecognitiveor
TheWienerdistribution(i.e.thediffusionmodel)usedinthis neural process is desired, as was in this paper), a whole host of
study does not incorporate trial-to-trial variability in drift rates single-trialEEGmeasurescouldbeincludedinahierarchicalmodel

M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130 129
| Table3                                                              |     |     |     |     |     |     |      | 4.4. Neurocognitivemodels |     |     |     |     |     |     |     |
| ------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
| Percentageofvarianceacrosssubjectsexplainedbyin-sampleprediction(R2 |     |     |     |     |     |     | )for |                           |     |     |     |     |     |     |     |
pred
summarystatisticsofknownsubjects’accuracy-RTdistributions.Allthreemodels
Theterm‘‘neurocognitive’’hasbeenusedtodescribetherecent
fitaccuracyandcorrect-RTt1dataverywell,explainingover92%ofmediancorrect-
RTandover90%ofaccuracyineachcondition.Howevernoneofthemodelsexplain trendofcombiningmathematicalbehavioralmodelsandobserva-
incorrect-RTt0distributionswell,aknownproblemforsimplediffusionmodels tionsofbrainbehaviortoexplainandpredictperceptualdecision
thatcanbeovercomebyincludingvariabledriftratesdirectlyinthelikelihood making(Palmeri,Schall,&Logan,2014).Theusefulnessofcombin-
function(Ratcliff,1978;Ratcliff&McKoon,2008). ing behavioral models and neural dynamics has been motivated
Predictionoftrainingdatafromknownsubjects ontheoreticalgrounds.Behavioralmodelssuggestlinksbetween
subjectbehaviorandcognitionwhilelaboratoryobservationand
|     |     |     | Model1 |     | Model2 | Model3 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ------ | --- | ------ | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Comparison EEG-δ,τ EEG-δ,τ,ς neuroimagingcansuggestlinksbetweenneuraldynamicsandcog-
nition.Thecombinationofthesemethodsthenprovidesapredic-
|     | 25tht1percentile |     |     | 96.5% | 97.1% | 97.8% |     |     |     |     |     |     |     |     |     |
| --- | ---------------- | --- | --- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t1median 96.6% 96.5% 97.3% tive chain of neural dynamics, cognition, and behavior. Another
| Low | 75tht1percentile |     |     | 91.4% | 93.4% | 94.5% |     |     |     |     |     |     |     |     |     |
| --- | ---------------- | --- | --- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
obviousbenefitistheinferencegainwhenpredictingmissingdata.
|     |     | Accuracy |     | 95.1% | 95.2% | 97.3% |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thatis,wewillbeabletobetterpredictbehaviorwhenbrainac-
|     |     | t0median | −118.8% |     | −108.3% | −111.6% |     |                      |     |         |            |           |       |            |     |
| --- | --- | -------- | ------- | --- | ------- | ------- | --- | -------------------- | --- | ------- | ---------- | --------- | ----- | ---------- | --- |
|     |     |          |         |     |         |         |     | tivity is available. |     | This is | especially | true when | using | hierarchi- |     |
25tht1percentile 86.0% 87.5% 88.6% cal Bayesian models as they maintain uncertainty in estimates
t1median 95.9% 95.6% 96.3% throughdifferentlevelsoftheanalysis(Turneretal.,2013;Van-
Medium 75tht1percentile 84.7% 89.4% 90.1% dekerckhove et al., 2011). While there are a variety of methods
|     |     | Accuracy |         | 90.7% | 94.1%   | 95.3%   |     |                     |        |                             |      |                      |     |         |       |
| --- | --- | -------- | ------- | ----- | ------- | ------- | --- | ------------------- | ------ | --------------------------- | ---- | -------------------- | --- | ------- | ----- |
|     |     |          |         |       |         |         |     | using cognitive     | models | to                          | find | cognitive correlates |     | in the  | brain |
|     |     | t0median | −163.9% |       | −158.6% | −163.9% |     |                     |        |                             |      |                      |     |         |       |
|     |     |          |         |       |         |         |     | dynamics(seeTurner, |        | Forstmann,Love,Palmeri,&Van |      |                      |     | Maanen, |       |
25tht1percentile 85.4% 87.3% 86.7% 2016, for a review of these methods), some studies do not fur-
|      |                  | t1median |     | 93.1% | 92.5% | 92.9% |     |                |     |           |        |              |     |       |        |
| ---- | ---------------- | -------- | --- | ----- | ----- | ----- | --- | -------------- | --- | --------- | ------ | ------------ | --- | ----- | ------ |
|      |                  |          |     |       |       |       |     | ther constrain | the | cognitive | models | by informing |     | those | models |
| High | 75tht1percentile |          |     | 79.1% | 83.8% | 84.0% |     |                |     |           |        |              |     |       |        |
Accuracy 95.9% 97.4% 95.9% withknownneurallinkstospecificcognitiveprocesses.Inthispa-
t0median −73.4% −71.2% −76.4% per,andourpreviousstudyofindividualdifferences(Nunezetal.,
|     |     |     |     |     |     |     |     | 2015), we | demonstrate | another |     | important | use of | neural | data in |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ----------- | ------- | --- | --------- | ------ | ------ | ------- |
cognitivemodels.Independentneuralmeasuresofcognitivepro-
| of decision | making, | using      | perhaps   | a           | simpler | model of decision |      |              |               |                |        |                |            |            |     |
| ----------- | ------- | ---------- | --------- | ----------- | ------- | ----------------- | ---- | ------------ | ------------- | -------------- | ------ | -------------- | ---------- | ---------- | --- |
|             |         |            |           |             |         |                   |      | cesses, such | as attention, |                | can be | used to better | understand |            | how |
| making      | such as | the linear | ballistic | accumulator |         | model to          | ease |              |               |                |        |                |            |            |     |
|             |         |            |           |             |         |                   |      | cognition    | influences    | the mechanisms |        | of behavior,   |            | furthering | ex- |
analysis(e.g.Forstmannetal.,2008;Ho,Brown,&Serences,2009;
planationandpredictionofthecognitiveprocess.
Rodriguez,Turner,VanZandt,&McClure,2015;vanMaanenetal.,
| 2011) or | a more | complicated | model | to  | improve | prediction. | The |     |     |     |     |     |     |     |     |
| -------- | ------ | ----------- | ----- | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
set of single-trial measures could include: ERP-like components Dataandcodesharing
| as we | discussed | in this | paper, | measures | of  | evoked amplitudes |     |     |     |     |     |     |     |     |     |
| ----- | --------- | ------- | ------ | -------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
in certain frequency bands, and measures of steady-state visual Pre-calculated EEG measures, raw behavioral data, MATLAB
evokedpotentials. stimuluscode,JAGScode,andanexamplesingle-trialEEGRscript
However,asobservedinTable3,wehavenotshownprediction areavailableuponrequestandinthefollowingrepository(asof
of reaction times for when subjects committed errors because February2016)iftheiruseisproperlycited.
(1) few errors were committed by the subjects in the presented https://github.com/mdnunez/mcntoolbox/.
| data and | (2) the | type | of model | we  | used has | been shown | to  |     |     |     |     |     |     |     |     |
| -------- | ------- | ---- | -------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
explainincorrect-RTdistributionswellonlywhenintrinsictrial-
Acknowledgments
to-trialvariabilityinevidenceaccumulationrates(asopposedto
extrinsicduetoneuralregressors,forwhichithasnotbeenshown)
is included in the likelihood function (Ratcliff, 1978; Ratcliff & Special thanks to Josh Tromberg and Aishwarya Gosai for
McKoon,2008).Infutureworkweplantocomparethepredictive help with EEG collection and artifact removal. Cort Horton is
abilityofmorecomplicatedmodelscontainingbothintrinsictrial- well appreciated for creating useful MATLAB functions for EEG
to-trialvariabilityandextrinsictrial-to-trialvariabilityinevidence visualization and artifact removal. Siyi Deng and Sam Thorpe
accumulationrateduetoexternalneuralmeasures. are thanked for their contributions to the FEM solutions and
The prediction ability of the presented models for accuracy anatomicalfMRIimagegeneration.Theanonymousreviewersare
and correct-RT distributions may have implications for Brain– appreciatedfortheirconstructivesuggestionsandcomments.
| Computer | Interface | (BCI) | frameworks, |     | especially | in paradigms |     |     |     |     |     |     |     |     |     |
| -------- | --------- | ----- | ----------- | --- | ---------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Funding
| which | attempt | to enhance | a   | participant’s |     | visual attention | to  |         |         |           |     |              |              |     |     |
| ----- | ------- | ---------- | --- | ------------- | --- | ---------------- | --- | ------- | ------- | --------- | --- | ------------ | ------------ | --- | --- |
|       |         |            |     |               |     |                  |     | MDN and | RS were | supported |     | by NIH grant | 2R01MH68004. |     |     |
particulartasktoimprovereactiontime.Tomaximizeprediction,
|       |               |     |         |        |     |               |     | JV was supported |     | by National | Science | Foundation |     | (NSF) | grants |
| ----- | ------------- | --- | ------- | ------ | --- | ------------- | --- | ---------------- | --- | ----------- | ------- | ---------- | --- | ----- | ------ |
| every | EEG attention |     | measure | should | be  | included, and | the |                  |     |             |         |            |     |       |        |
#1230118and#1534472fromtheMethods,Measurements,and
| predictors     | that       | offer the | best        | out-of-sample |           | prediction       | within |            |           |       |        |      |          |           |     |
| -------------- | ---------- | --------- | ----------- | ------------- | --------- | ---------------- | ------ | ---------- | --------- | ----- | ------ | ---- | -------- | --------- | --- |
|                |            |           |             |               |           |                  |        | Statistics | panel and | grant | #48192 | from | the John | Templeton |     |
| initial K-fold | validation |           | sets should | be            | included. | After collection |        |            |           |       |        |      |          |           |     |
Foundation.
| of behavior | and | EEG from | a few | participants |     | and a hierarchical |     |     |     |     |     |     |     |     |     |
| ----------- | --- | -------- | ----- | ------------ | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Bayesian analysis of the data, later participants’ single-trial Conflictofintereststatement
preprocessing times or evidence accumulation rates could be The authors declare that the research was conducted in the
predicted using only single-trial EEG measures. Conceivably this absenceofanycommercialorfinancialrelationshipsthatcouldbe
| would | allow for | trial-by-trial |     | intervention | in  | order to enhance |     |     |     |     |     |     |     |     |     |
| ----- | --------- | -------------- | --- | ------------ | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
construedasapotentialconflictofinterest.
| a participant’s | attention |            | during  | the task,   | perhaps | using           | neural |            |                       |     |     |     |     |     |     |
| --------------- | --------- | ---------- | ------- | ----------- | ------- | --------------- | ------ | ---------- | --------------------- | --- | --- | --- | --- | --- | --- |
| feedback        | (e.g.     | via direct | current | stimulation |         | or transcranial |        |            |                       |     |     |     |     |     |     |
|                 |           |            |         |             |         |                 |        | AppendixA. | Supplementarymaterial |     |     |     |     |     |     |
magneticstimulation)duringthosetrialsinwhichaparticipantis
predictedtobeslowintheirresponsebecauseofasmallevidence
accumulation rate (e.g. when an N200 magnitude is small) or The supplementary material contains posterior predictive
a slow preprocessing time (e.g. when an N200 latency is long). coverage plots of correct-RT distributions for each subject and
Althoughwhethertheparticipantcouldusesuchfeedbackintime condition. Supplementary material related to this article can be
toaffectreactiontimeandaccuracyremainstobetested. foundonlineathttp://dx.doi.org/10.1016/j.jmp.2016.03.003.

130 M.D.Nunezetal./JournalofMathematicalPsychology76(2017)117–130
References Nunez,M.D.,Srinivasan,R.,&Vandekerckhove,J.(2015).Individualdifferencesin
attentioninfluenceperceptualdecisionmaking.FrontiersinPsychology,6.
Bridwell,D.A.,Hecker,E.A.,Serences,J.T.,&Srinivasan,R.(2013).Individual O’Connell,R.G.,Dockree,P.M.,&Kelly,S.P.(2012).Asupramodalaccumulation-
differencesinattentionstrategiesduringdetection,finediscrimination,and to-bound signal that determines perceptual decisions in humans. Nature
coarsediscrimination.JournalofNeurophysiology,110,784–794. Neuroscience,15,1729–1735.
Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice Palmeri, T. J., Schall, J. D., & Logan, G. D. (2014). Neurocognitive modeling
responsetime:linearballisticaccumulation.CognitivePsychology,57,153–178. of perceptual decision making. In Oxford handbook of computational and
Brown,S.D.,Ratcliff,R.,&Smith,P.L.(2006).Evaluatingmethodsforapproximating mathematicalpsychology.
stochasticdifferentialequations.TheJournalofMathematicalPsychology,50, Parra,L.C.,Spence,C.D.,Gerson,A.D.,&Sajda,P.(2005).Recipesforthelinear
402–410. analysisofEEG.Neuroimage,28,326–341.
Buschman, T. J., & Miller, E. K. (2007). Top-down versus bottom-up control Philiastides,M.G.,Heekeren,H.R.,&Sajda,P.(2014).Humanscalppotentialsreflect
of attention in the prefrontal and posterior parietal cortices. Science, 315, amixtureofdecision-relatedsignalsduringperceptualchoices.TheJournalof
1860–1862. Neuroscience,34,16877–16889.
Buzsaki,G.(2006).Rhythmsofthebrain.OxfordUniversityPress. Philiastides, M. G., & Sajda, P. (2006). Temporal characterization of the neural
Calhoun,V.,Adali,T.,Pearlson,G.,&Pekar,J.(2001).Amethodformakinggroup correlatesofperceptualdecisionmakinginthehumanbrain.CerebralCortex,
inferencesfromfunctionalMRIdatausingindependentcomponentanalysis. 16,509–518.
HumanBrainMapping,14,140–151. Philiastides,M.G.,&Sajda,P.(2007).EEG-informedFMRIrevealsspatiotemporal
Callaway,E.,&Halliday,R.(1982).Theeffectofattentionaleffortonvisualevoked characteristicsofperceptualdecisionmaking.TheJournalofNeuroscience,27,
potentialn1latency.PsychiatryResearch,7,299–308. 13082–13091.
Corbetta,M.,&Shulman,G.L.(2002).Controlofgoal-directedandstimulus-driven Plummer,M.,etal.(2003).JAGS:AprogramforanalysisofBayesiangraphical
attentioninthebrain.NatureReviewsNeuroscience,3,201–215. modelsusingGibbssampling.InProceedingsofthe3rdinternationalworkshopon
Davis, E. T., & Graham, N. (1981). Spatial frequency uncertainty effects in the distributedstatisticalcomputing,Vol.124(p.125).TechnischeUniversitatWien.
detectionofsinusoidalgratings.VisionResearch,21,705–712. Pommier,J.,&Renard,Y.(2005).Getfem++,anopensourcegenericc++libraryfor
Deng,S.,Winter,W.,Thorpe,S.,&Srinivasan,R.(2012).Improvedsurfacelaplacian finiteelementmethods.
estimatesofcorticalpotentialusingrealisticmodelsofheadgeometry.IEEE Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85,59.
TransactionsonBiomedicalEngineering,59,2979–2985. Ratcliff,R.,&McKoon,G.(2008).Thediffusiondecisionmodel:theoryanddatafor
Desimone,R.,&Duncan,J.(1995).Neuralmechanismsofselectivevisualattention. two-choicedecisiontasks.NeuralComputation,20,873–922.
AnnualReviewofNeuroscience,18,193–222. Ratcliff,R.,Philiastides,M.G.,&Sajda,P.(2009).Qualityofevidenceforperceptual
Dosher,B.A.,&Lu,Z.-L.(2000).Noiseexclusioninspatialattention.Psychological decisionmakingisindexedbytrial-to-trialvariabilityoftheEEG.Proceedings
Science,11,139–146. oftheNationalAcademyofSciences,106,6539–6544.
Dosher,B.A.,&Lu,Z.-L.(2000).Mechanismsofperceptualattentioninprecuingof Rodriguez, C. A., Turner, B. M., Van Zandt, T., & McClure, S. M. (2015). The
location.VisionResearch,40,1269–1292. neuralbasisofvalueaccumulationinintertemporalchoice.EuropeanJournal
Dutilh, G., Vandekerckhove, J., Forstmann, B. U., Keuleers, E., Brysbaert, M., & ofNeuroscience,42,2179–2189.
Wagenmakers,E.-J.(2012).Testingtheoriesofpost-errorslowing.Attention Ross,S.M.(2014).Introductiontoprobabilitymodels.Academicpress.
Perception&Psychophysics,74,454–465. Shadlen,M.N.,&Newsome,W.T.(1996).Motionperception:seeinganddeciding.
Eriksen, C. W., & Hoffman, J. E. (1972). Temporal and spatial characteristics ProceedingsoftheNationalAcademyofSciences,93,628–633.
of selective encoding from visual displays. Perception & Psychophysics, 12, Shadlen,M.N.,&Newsome,W.T.(2001).Neuralbasisofaperceptualdecisionin
201–204. theparietalcortex(arealip)oftheRhesusmonkey.JournalofNeurophysiology,
Fischl,B.,vanderKouwe,A.,Destrieux,C.,Halgren,E.,Ségonne,F.,Salat,D.H.,etal.
86,1916–1936.
(2004).Automaticallyparcellatingthehumancerebralcortex.CerebralCortex,
Shaw, M. L., & Shaw, P. (1977). Optimal allocation of cognitive resources to
14,11–22.
spatial locations. Journal of Experimental Psychology: Human Perception and
Forstmann,B.U.,Dutilh,G.,Brown,S.,Neumann,J.,VonCramon,D.Y.,Ridderinkhof,
Performance,3,201.
K.R.,etal.(2008).Striatumandpre-SMAfacilitatedecision-makingundertime
Smith,P.L.,&Ratcliff,R.(2009).Anintegratedtheoryofattentionanddecision
pressure.ProceedingsoftheNationalAcademyofSciences,105,17538–17542.
makinginvisualsignaldetection.PsychologicalReview,116,283.
Gelman,A.,&Rubin,D.B.(1992).Inferencefromiterativesimulationusingmultiple
Spieler,D.H.,Balota,D.A.,&Faust,M.E.(2000).Levelsofselectiveattentionre-
sequences.StatisticalScience,457–472.
vealedthroughanalysesofresponsetimedistributions.JournalofExperimental
Harter,M.R.,&Aine,C.J.(1984).Brainmechanismsofvisualselectiveattention.
Psychology:HumanPerceptionandPerformance,26,506.
VarietiesofAttention,293–321.
Ho, T. C., Brown, S., & Serences, J. T. (2009). Domain general mechanisms of Stone,M.(1960).Modelsforchoice-reactiontime.Psychometrika,25,251–260.
perceptualdecisionmakinginhumancortex.TheJournalofNeuroscience,29, Tange, O. (2011). GNU parallel—The command-line power tool. The USENIX
8675–8687. Magazine,36,42–47.
Hong,B.,Guo,F.,Liu,T.,Gao,X.,&Gao,S.(2009).N200-spellerusingmotion-onset Turner, B. M., Forstmann, B. U., Love, B. C., Palmeri, T. J., & Van Maanen, L.
visualresponse.ClinicalNeurophysiology,120,1658–1666. (2016).Approachestoanalysisinmodel-basedcognitiveneuroscience.Journal
Jung,T.-P.,Makeig,S.,Humphries,C.,Lee,T.-W.,Mckeown,M.J.,Iragui,V.,etal. ofMathematicalPsychology,.
(2000).Removingelectroencephalographicartifactsbyblindsourceseparation. Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.-J.,Brown,S.D.,Sederberg,P.B.,&
Psychophysiology,37,163–178. Steyvers,M.(2013).ABayesianframeworkforsimultaneouslymodelingneural
Krishnan, L., Kang, A., Sperling, G., & Srinivasan, R. (2013). Neural strategies andbehavioraldata.NeuroImage,72,193–206.
for selective attention distinguish fast-action video game players. Brain Turner,B.M.,vanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
Topography,26,83–97. stractionsthroughneuroimaging:Theneuraldriftdiffusionmodel.Psychologi-
Lee,T.-W.,Girolami,M.,&Sejnowski,T.J.(1999).Independentcomponentanalysis calReview,122,312.
usinganextendedinfomaxalgorithmformixedsubgaussianandsupergaussian Twomey,D.M.,Murphy,P.R.,Kelly,S.P.,&O’Connell,R.G.(2015).Theclassic
sources.NeuralComputation,11,417–441. p300 encodes a build-to-threshold decision variable. European Journal of
Lee,M.D.,&Wagenmakers,E.-J.(2014).Bayesiancognitivemodeling:Apractical Neuroscience,.
course.CambridgeUniversityPress. Vandekerckhove,J.,&Tuerlinckx,F.(2007).Fittingtheratcliffdiffusionmodelto
Lu,Z.-L.,&Dosher,B.A.(1998).Externalnoisedistinguishesattentionmechanisms. experimentaldata.PsychonomicBulletin&Review,14,1011–1026.
VisionResearch,38,1183–1198. Vandekerckhove,J.,Tuerlinckx,F.,&Lee,M.D.(2011).Hierarchicaldiffusionmodels
Luck,S.J.,Woodman,G.F.,&Vogel,E.K.(2000).Event-relatedpotentialstudiesof fortwo-choiceresponsetimes.PsychologicalMethods,16,44.
attention.TrendsinCognitiveSciences,4,432–440. vanMaanen,L.,Brown,S.D.,Eichele,T.,Wagenmakers,E.-J.,Ho,T.,Serences,J.,etal.
Makeig, S., Bell, A. J., Jung, T.-P., Sejnowski, T. J., et al. (1996). Independent (2011).Neuralcorrelatesoftrial-to-trialfluctuationsinresponsecaution.The
component analysis of electroencephalographic data. Advances in Neural JournalofNeuroscience,31,17488–17495.
InformationProcessingSystems,145–151. vanVugt,M.K.,Simen,P.,Nystrom,L.E.,Holmes,P.,&Cohen,J.D.(2012).EEG
Meister,M.L.,Hennig,J.A.,&Huk,A.C.(2013).Signalmultiplexingandsingle- oscillations reveal neural correlates of evidence accumulation. Frontiers in
neuroncomputationsinlateralintraparietalareaduringdecision-making.The Neuroscience,6.
JournalofNeuroscience,33,2254–2267. Voss,A.,Rothermund,K.,&Voss,J.(2004).Interpretingtheparametersofthe
Mulder, M., Van Maanen, L., & Forstmann, B. (2014). Perceptual decision diffusionmodel:Anempiricalvalidation.Memory&Cognition,32,1206–1220.
neurosciences—Amodel-basedreview.Neuroscience,277,872–884. Wabersich,D.,&Vandekerckhove,J.(2014).ExtendingJAGS:atutorialonadding
Nunez,P.L.,&Pilgreen,K.L.(1991).Thespline-laplacianinclinicalneurophysiol- custom distributions to JAGS (with a diffusion model example). Behavior
ogy:amethodtoimproveEEGspatialresolution.JournalofClinicalNeurophys- ResearchMethods,46,15–28.
iology,8,397–413. White,C.N.,Congdon,E.,Mumford,J.A.,Karlsgodt,K.H.,Sabb,F.W.,Freimer,
Nunez,P.,Silberstein,R.,Cadusch,P.,Wijesinghe,R.,Westdorp,A.,&Srinivasan, N. B., London, E. D., Cannon, T. D., Bilder, R. M., & Poldrack, R. A. (2014).
R.(1994).AtheoreticalandexperimentalstudyofhighresolutionEEGbased Decomposing decision components in the stop-signal task: a model-based
onsurfacelaplaciansandcorticalimaging.ElectroencephalographyandClinical approachtoindividualdifferencesininhibitorycontrol.JournalofCognitive
Neurophysiology,90,40–57. Neuroscience,26,1601–1614.
Nunez,P.L.,&Srinivasan,R.(2006).Electricfieldsofthebrain:Theneurophysicsof Wiecki,T.V.,Sofer,I.,&Frank,M.J.(2013).HDDM:hierarchicalBayesianestimation
EEG.OxfordUniversityPress. ofthedrift-diffusionmodelinpython.FrontiersinNeuroinformatics,7.