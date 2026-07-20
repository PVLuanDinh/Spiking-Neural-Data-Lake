JournalofMathematicalPsychology76(2017)172–183
ContentslistsavailableatScienceDirect
|     |     |     | Journal |     | of Mathematical |     |     | Psychology |     |     |     |     |     |     |     |
| --- | --- | --- | ------- | --- | --------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
journalhomepage:www.elsevier.com/locate/jmp
| A primer | on encoding |     | models |     | in sensory |     | neuroscience |     |     |     |     |     |     |     |     |
| -------- | ----------- | --- | ------ | --- | ---------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
MarcelA.J.vanGerven
RadboudUniversity,DondersInstituteforBrain,CognitionandBehaviour,Montessorilaan3,6525HR,Nijmegen,TheNetherlands
| h i g h | l i g h t s |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
• Anoverviewofcomputationalmodelinginsensoryneuroscienceisprovided.
• Thetheoreticalbackgroundandapplicationofencodingmodelsaredescribed.
•
Futureresearchdirectionsontheuseofencodingmodelsinsensoryneurosciencearediscussed.
| a r t i c | l e i n f o |     |     | a b s t | r a c | t   |     |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Articlehistory: Aprincipalgoalinsensoryneuroscienceistounderstandhowpropertiesofourenvironmentarereflected
Availableonline18July2016
inneuralactivitypatterns.Recentadvancesincomputationalmodelingprovideincreasinglyaccurate
predictionsofhowneuralpopulationsacrossthebrainrespondtocomplexnaturalisticstimuli.Theem-
Keywords: ployedcomputationalmodels,referredtoasencodingmodels,explicitlytransformcomplexstimuliinto
Populationreceptivefield
observedneuralresponses.Thisrapidlydevelopingfieldisbecomingincreasinglyimportantinsensory
Encoding
neuroscienceasitprovidesdetailedinsightsintothefunctionalorganizationofneuralrepresentations.
Decoding
Thepresentworkstartsbydiscussingthetheoreticalunderpinningsofencodingmodels.Next,variousap-
Sensoryneuroscience
plicationsofencodingmodelsarereviewed.Finally,potentialresearchdirectionsthatmayshapefuture
workinthisareaofresearcharedescribed.
©2016ElsevierInc.Allrightsreserved.
1. Introduction notionofthereceptivefield(RF),definedasthatportionof(concrete
orabstract)sensoryspacewhichelicitsaneuronalresponsewhen
A fundamental question in neuroscience is how the human stimulated(Sherrington,1906),thatsensoryneurosciencewasput
brainmakessenseofitsenvironment.Howdoesthebrainmanage on a more quantitative footing. Let x = ( x ,..., x )T be the
|     |     |     |     |     |     |     |     |     |     |     |     |     | 1   | N   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
to learn about, represent and recognize statistical invariances in sensory input. The receptive field of a neuron is a filter w =
the environment that guide its actions, ultimately ensuring our (w ,...,w )Tsuchthat
|             |              |       |              |       |          |     | 1   |     | N   |     |     |     |     |     |     |
| ----------- | ------------ | ----- | ------------ | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| survival in | a world that | is in | a continuous | state | of flux? | The |     |     |     |     |     |     |     |     |     |
|             |              |       |              |       |          |     |     |   |     |     |     |     |     |     |     |
representation of invariant features of the environment is the r =g wTx (1)
subjectmatterofsensoryneuroscience.Inrecentyears,advances
|                                                         |     |     |     |     |     |     | gives                                         | the (expected) |     | instantaneous |     | firing | rate | of that      | neuron |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --------------------------------------------- | -------------- | --- | ------------- | --- | ------ | ---- | ------------ | ------ |
| incomputationalmodelinghavechangedthewaysensorycortical |     |     |     |     |     |     |                                               |                |     |               |     |        |      | (·)          |        |
|                                                         |     |     |     |     |     |     | (Gerstner,Kistler,Naud,&Paninski,2014).Here,g |                |     |               |     |        |      | isanonlinear |        |
responsestoexternalstimuliareinvestigated.Theobjectiveofthe
functionthatmapsitsinputtoafiringrate.Fig.1givesanexample
presentworkistoprovideanoverviewoftheseadvancesandto
wherethestimulusxisgivenbyavisualimageandthereceptive
pointoutnewdirectionsinthisexcitingfieldofresearch.Inthe
fieldwisgivenbyaGaussiankernel.Notethateachimagepixelx
| following,anumberofconceptsareintroducedthatareattheroot |     |     |     |     |     |     |                                              |        |               |     |               |          |            | w      | i       |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | ------ | ------------- | --- | ------------- | -------- | ---------- | ------ | ------- |
|                                                          |     |     |     |     |     |     | isassociatedwithauniquereceptivefieldelement |        |               |     |               |          |            | .      |         |
| ofcomputationalmodelinginsensoryneuroscience.            |     |     |     |     |     |     |                                              |        |               |     |               |          |            | i      |         |
|                                                          |     |     |     |     |     |     | In                                           | visual | neuroscience, |     | the receptive |          | field has  | become | a key   |
|                                                          |     |     |     |     |     |     | concept                                      | for    | understanding |     | the           | response | properties | of     | neurons |
1.1. Receptivefields along the visual pathway (Barlow, 1953; Hartline, 1940; Hubel
|     |     |     |     |     |     |     | & Wiesel, |     | 1959; Kuffler, |     | 1953). | While | the characterization |     | of  |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | -------------- | --- | ------ | ----- | -------------------- | --- | --- |
Thescientificstudyofsensationandperceptionwasinitiated
receptivefieldpropertieshaspredominantlyfocusedonthevisual
byWundt(1862).However,itwasnotuntiltheintroductionofthe
|                                          |     |     |     |     |     |     | modality  | (Ringach,        |         | 2004), | similar   | successes | have     | been          | booked  |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | ---------------- | ------- | ------ | --------- | --------- | -------- | ------------- | ------- |
|                                          |     |     |     |     |     |     | in the    | characterization |         | of     | auditory, | olfactory | and      | somatosensory |         |
|                                          |     |     |     |     |     |     | receptive | fields           | (Hsiao, | 2008;  | Knudsen   | &         | Konishi, | 1978;         | Wilson, |
| E-mailaddress:m.vangerven@donders.ru.nl. |     |     |     |     |     |     | 2001).    |                  |         |        |           |           |          |               |         |
http://dx.doi.org/10.1016/j.jmp.2016.06.009
0022-2496/©2016ElsevierInc.Allrightsreserved.

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 173
Fig.1. Animagewhosepixelsmakeusthestimulusvectorxandwhosereceptivefieldwisgivenby(avectorizedrepresentationof)anisotropicGaussianlocatedinthe
upperrightvisualfield.TheresponseisgeneratedbyapplyinganonlinearfunctiongtotheneuronalinputwTx.Theresponseofoneneuronristhespikecountwithinsome
timewindow(redrectangle).Theresponseofapoolofneuronsisthepopulationactivitya,definedintermsofthenumberofspikesthatoccurwithinsomesmalltime
windowacrossallneuronsinthepool(greenrectangle).(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthis
article.)
Theuseofreceptivefieldsasatheoreticalconstructhasbeen WerefertoEq.(2)asapopulationreceptivefieldmodel,mapping
extraordinarily successful in sensory neuroscience. At the same asensorystimulustoapopulationresponse(cf.Fig.1).
time,accuratecharacterizationofthereceptivefieldsofindividual Onemightquestionwhyoneembarksonpopulationreceptive
neuronsremainsextremelydifficultforanumberofreasons.First, fieldmodelinginthefirstplacegiventhat,incontrasttoanalyzing
RFstructurecanbehardtocharacterize(Carandinietal.,2005), single-neuronresponses,itdoesnotleadtoadirectunderstanding
particularly in downstream areas (Gallant, Braun, & Van Essen, ofneuralcoding.Animportantmotivationisthatunderstanding
1993;Okazawa,Tajima,&Komatsu,2015).Second,RFproperties how population responses are organized at a macroscopic level
are modulated by non-classical receptive field effects such as may provide insight into the functional organization of neural
surroundsuppression(Fitzpatrick,2000;Joo,Boynton,&Murray, representations in the human brain. This, in turn, may guide
2012; Smith & Muckli, 2010; Zipser, Lamme, & Schiller, 1996).
hypothesis formation on the computational principles by which
Third, they are modulated by top-down processes such as task
such representations emerge. Research suggests that population
demand (Emadi & Esteky, 2014; McKee, Riesenhuber, Miller, &
responses reflect computational principles operating at a lower
Freedman,2014)andattention(Çukur,Nishimoto,Huth,&Gallant,
scale(Kamitani&Tong,2005).However,whetherornotthiscan
2013;David,Hayden,Mazer,&Gallant,2008).Theseconfounding
beexplainedbylarge-scalebiasesthatexistatamacroscopiclevel
factors make testing of RF properties in controlled experimental
hasbeenvigorouslydebated(e.g.Carlson,2014;Cichy&Pantazis,
settingsdifficult.
2015;Freeman,Brouwer,Heeger,&Merriam,2011;Kriegeskorte,
Cusack, & Bandettini, 2010; Op de Beeck, 2010; Swisher et al.,
1.2. Populationreceptivefields
2010).
A receptive field characterizes what stimulus properties a
singleneuronrespondsto.However,ifweconsidernon-invasive 1.3. Encodingmodels
measurement techniques used to probe human brain function
then we can only measure the aggregate activity of a pool of In general, one may consider multiple neuronal populations.
neurons. For example, functional magnetic resonance imaging Theactivityineachofthesepopulationsmayinfluencehowsignals
(fMRI) measures the blood oxygenation-level dependent (BOLD) aremeasuredbysensorsusingavarietyofrecordingtechniques.
response across thousands of volume elements (voxels) in the Anencodingmodelseekstoexplain(1)howastimulusmodulates
brain. Each voxel captures the hemodynamic response induced the activity of p neuronal populations and (2) how population
byaneuronalpopulationconsistingofabout5.5millionneurons activity affects data recorded from q sensors (Kriegeskorte &
(Logothetis, 2008). As another example, electroencephalography Kreiman,2011;Naselaris,Kay,Nishimoto,&Gallant,2011).1
(EEG)andmagnetoencephalography(MEG)measuretheelectrical Consideracollectionofpneuronalpopulations.Eachofthese
and magnetic fields, respectively, that are generated by the populationscanbesensitivetodifferentstimulusfeatures.Thatis,
(weighted)responsesofmultiplepopulationsofneuronsdispersed weassumethatthepopulationresponsestoastimuluspresented
acrossthebrainvolume. attimetaregivenby
Forapopulationofneurons,thetermpopulationreceptivefield
(pRF) was introduced to refer to those stimulus properties that a t =g ( x t , a t−1 ). (3)
drivetheirpooledresponse(Victor,Purpura,Katz,&Mao,1994).In whereg=(
g
,...,
g
)Tstandsforthenonlineartransformations
1 p
caseoffMRI,whichwillbethemainfocusofthisreview,thepRFis
thatarespecifictoeachneuronalpopulation.Thisequationmakes
definedatthelevelofsinglevoxelsandalsoreferredtoasthevoxel
receptivefield(Kriegeskorte&Kreiman,2011).Analogoustothe
receptivefieldofanindividualneuron,apopulationreceptivefield
isafilterwsuchthatthe(expected)populationactivityisgivenby 1 Here,sensorsaretakentostandforeithervoxel-specificBOLDmeasuresincase
a=g  wTx . (2) o o f r f i M nt R ra I, c e r x a t n r i a a c ll r y an m ia e l a ly su m re e d as m ur u e l d ti- e u le n c it tr a ic ct o iv r i m ty a . gneticfieldsincaseofEEGandMEG,

174 M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183
explicit that the population activity at some time t depends reportresultsforahandfulofindividuals,asiscommoninanimal
nonlinearlyonthestimuluspresentedattimet aswellasonthe studies,wheredatacollectionislaboriousaswell,albeitforother
populationactivityatthepreviouspointintime.2 reasons.3 Afterestimatingtheparametersofanencodingmodel,
Note that Eq. (3) is a more general formulation than Eq. (2). itcanbeusedtopredicthowstimuliinducemeasuredbrainre-
Eq. (2) assumes that the population activity is given by a lin- sponses(Victoretal.,1994).
ear–nonlinearmodel,whichisknowntoyieldpoorpredictionsin
caseofcomplexnaturalisticstimuli(Frégnac&Bathellier,2015), 2.1. Parameterestimation
whereasEq.(3)isgivenbyanarbitrarynonlinearfunctionofthe
| (potentially | complete) | stimulus | history | through | the | dependence |     |     |     |     |     |     |     |     |
| ------------ | --------- | -------- | ------- | ------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
Givennuniquepresentedstimuliandmobservedmeasurement
onthepastpopulationactivity.ItfollowsthatpRFmodelsaresub-
|     |     |     |     |     |     |     | vectors, | anencoding | model | seeksto | predict | theset | of measure- |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ----- | ------- | ------- | ------ | ----------- | --- |
sumedbyencodingmodels. ments{y }asafunctionofthesetofsensoryinputs{x }usinga
|                                                       |     |     |     |     |     |     |                      | t   |     |                                     |     |     | t   |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | ----------------------------------- | --- | --- | --- | --- |
| Anencodingmodelalsomakesexplicithowpopulationactivity |     |     |     |     |     |     |                      |     | φ   |                                     |     |     |     |     |
|                                                       |     |     |     |     |     |     | suitablefeaturespace |     |     | andforwardmodelsf.Assumingthat,over |     |     |     |     |
ismeasuredatthesensorlevel.Thesemeasurementsmaydepend
time,measurementsareconditionallyindependentgiventhepop-
onthehistoryofpopulationactivity,e.g.duetothelaginthehemo- ulationactivity,togetherwiththeassumeddeterministicdepen-
dynamicresponsewhencollectingfMRIBOLDdata.Toaccommo- denceofthefeaturespaceontheinputstimulus,wemaywrite4
dateforthis,let
|       |       |         |     |     |     |     | ({y | })=  |  ( | |φ ), |     |     |     |     |
| ----- | ----- | ------- | --- | --- | --- | --- | --- | ---- | --- | ----- | --- | --- | --- | --- |
|       |       |         |     |     |     |     | p   | }|{x | p y |       |     |     |     | (6) |
| φ =(a | , a   | ,..., a | )T  |     |     | (4) | t   | t    |     | t t   |     |     |     |     |
| t t−k | t−k+1 | t       |     |     |     |     |     |      | t   |       |     |     |     |     |
φ
denoteahistoryofneuralpopulationactivity.Wealsoreferto wheretrangesoverthetimesatwhichameasurementwastaken.
asafeaturespacesinceitrepresentsacollectionofstimulusfea- Itiscomputationallyexpedienttoassumethattheconditional
tures(nonlineartransformationsofthestimulus).Wenowdefine densitiesaregivenbyamultivariateGaussian
theexpectedmeasurementvectoras
|        |     |     |     |     |     |     | p ( y |φ     | )=N( | y ;µ ,(cid:54)),                          |     |     |     |     | (7) |
| ------ | --- | --- | --- | --- | --- | --- | ------------ | ---- | ----------------------------------------- | --- | --- | --- | --- | --- |
| µ (φ ) |     |     |     |     |     |     | t            | t    | t t                                       |     |     |     |     |     |
| =f     |     |     |     |     |     | (5) |              |      |                                           |     |     |     |     |     |
| t t    |     |     |     |     |     |     |              |      | µ                                         |     |     |     |     |     |
|        |     |     |     |     |     |     | wherethemean |      | t representstheexpectedvalueofthemeasure- |     |     |     |     |     |
where f = ( f ,..., f ) is a vector-valued function such that f nEq.(5),andthecovariancematrix(cid:54)captures
|     | 1   | q   |     |     |     | j   | ments,asdefinedi |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
theforwardmodelwhichmapsthe(lagged)responsesofpdistinct
theresidualmeasurementnoise.5Ifweassumethateachforward
poolsofneuronstothejthobservedresponse. modelisgivenbyalinearmappingthenEq.(6)canbeconveniently
| Summarizing,encodingmodelscanbethoughtofasproviding      |     |     |     |     |     |     | writtenas |          |     |      |            |     |     |     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | -------- | --- | ---- | ---------- | --- | --- | --- |
| acommonframeworkthatallowsinvestigating(highlynonlinear) |     |     |     |     |     |     |           |          |    |      |            |     |     |     |
|                                                          |     |     |     |     |     |     | ({y       | }|{x })= | N(  | ;BTφ | ,(cid:54)) |     |     |     |
response properties of different cortical areas as induced by p t t y t t (8)
| arbitrarysensoryinputs,whereresponsesaremeasuredindirectly |     |     |     |     |     |     |     |     | t   |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
via signal-altering neural recording methods. The present paper withB=β ,...,β 
amatrixofregressioncoefficients.
provides an overview of this new area of research and explores 1 q
|              |        |     |         |            |       |          | Eq. | (8) is | known as | a linearizing | encoding | model | since | it only |
| ------------ | ------ | --- | ------- | ---------- | ----- | -------- | --- | ------ | -------- | ------------- | -------- | ----- | ----- | ------- |
| how encoding | models | may | be used | to provide | novel | insights |     |        |          |               |          |       |       |         |
requiresalinearmappingbetweenthepopulationresponsesand
| into how neural | responses |     | are affected | by  | bottom-up | sensory |     |     |     |     |     |     |     |     |
| --------------- | --------- | --- | ------------ | --- | --------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
theobservations(Naselarisetal.,2011).Inthissetting,parameter
evidenceaswellastop-downexecutiveprocessesinnaturalistic
|     |     |     |     |     |     |     | estimation |     | entails solving | a   | set of multiple |     | linear regression |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --------------- | --- | --------------- | --- | ----------------- | --- |
paradigms. Section 2 describes how encoding models can be problems, yielding the regression coefficients B. Since φ can be
estimated and inverted using experimental data. Section 3 extremelyhigh-dimensionalandthestimulusfeaturescandisplay
addresses related approaches. In Section 4, the construction and highcollinearity,standardordinaryleastsquaresestimationofB
| use of different | feature | spaces | are | reviewed. | Section | 5 describes |     |     |     |     |     |     |     |     |
| ---------------- | ------- | ------ | --- | --------- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
isunstable.Insteadregularizedestimatorsshouldbeusedwhich
howencodingmodelsmaybeusedtoinformaboutthenatureof
imposeconstraintsonB,therebymakingtheparameterestimates
| neural representations. |     | Section | 6 provides | an  | outlook | on future |     |     |     |     |     |     |     |     |
| ----------------------- | --- | ------- | ---------- | --- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
morewell-behaved(Schoenmakers,Barth,Heskes,&vanGerven,
applicationsofencodingmodelsinsensoryneuroscience.Weend (cid:54)
|     |     |     |     |     |     |     | 2013; | Wu, David, | & Gallant, | 2006).6 | The | noise | covariance | can |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | ---------- | ------- | --- | ----- | ---------- | --- |
withsomegeneralconclusionsinSection7.
eitherbeanalyticallyapproximatedorestimatedfromaseparate
|     |     |     |     |     |     |     | dataset. | Validation | of  | an encoding | model | typically | proceeds | by  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | ----------- | ----- | --------- | -------- | --- |
2. Estimatingandinvertingencodingmodels computingthevarianceexplainedbythemodelonaseparatetest
setwhichhasnotbeenusedduringmodelestimation.
In contrast to more controlled paradigms in cognitive neuro- Eq.(8)subsumesmanyoftheencodingmodelsthathavebeen
science(Rust&Movshon,2005),encodingmodelstypicallyseek developed to date, such as Güçlü and van Gerven (2014, 2015),
| to characterize | how | arbitrary | sensory | input affects |     | observed re- |      |            |          |     |                 |            |     |          |
| --------------- | --- | --------- | ------- | ------------- | --- | ------------ | ---- | ---------- | -------- | --- | --------------- | ---------- | --- | -------- |
|                 |     |           |         |               |     |              | Kay, | Naselaris, | Prenger, | and | Gallant (2008), | Naselaris, |     | Prenger, |
sponses.Thisinputoftentakestheformofhigh-dimensionalnatu-
|     |     |     |     |     |     |     | Kay, | Oliver, | and Gallant | (2009) | and Schoenmakers |     | et al. | (2013). |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------- | ----------- | ------ | ---------------- | --- | ------ | ------- |
ralisticstimuli(Creutzfeldt&Nothdurft,1978;Einhäuser&König, Theessentialdifferencebetweenthesevariousencodingmodels
2010;Felsen&Dan,2005;Gallant,Connor,&VanEssen,1998;Has-
son,Malach,&Heeger,2010;Kayser,Körding,&König,2004).Nat-
uralisticstimulifeaturecharacteristicstatisticsthatareusuallynot
3
containedinhighlycontrolledstimuli(e.g.barsandgratings).To Onemightalsoarguethatthehighlydetailedresponsepropertiesascaptured
byencodingmodelscanbehighlyindividualized,thusmakinggroup-levelanalyses
learnhowthishigh-dimensionalstimulusspacemapsontoneural
lessmeaningful.
responses,asufficientnumberofstimulus–responsepairsneeds
4 Inreality,measurementscanbehighly(auto)correlatedevenafterconditioning
tobecollected,resultinginmanyhoursofscanningtimepersub-
onthepopulationactivity.Inpractice,thismaybecircumventedusingparticular
ject.Forthisreason,studiesthatemployencodingmodelstendto
preprocessingstepssuchasaveragingoverblocksorusingtrial-specificregression
coefficientsassurrogatemeasurements.
5 Weoftenleavemodelparametersimplicitinournotationtoreduceclutter.
6 Forexample,ridgeregressionassumesazero-meanGaussianprioronindi-
2 Oneshouldrealizethatthisformulationassumesuniformconductiondelays vidualcoefficients,therebymitigatingcollinearityissues.Theuseofregularization
betweenpopulationsandtheabsenceofresponsevariabilitywithinpopulations. comesattheexpenseofhavingtoestimatearegularizationparameterusingsepa-
| Often,thetimedependenceisomittedaswell. |     |     |     |     |     |     | ratevalidationdata. |     |     |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- |

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 175
| follows from | their       | use   | of different feature | spaces      | φ , reflecting |     |     |     |     |     |     |     |
| ------------ | ----------- | ----- | -------------------- | ----------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| different    | assumptions | about | how neuronal         | populations | respond        |     |     |     |     |     |     |     |
tosensoryinput.Inthissetting,regressioncoefficientsBareused
to(1)beabletopredictobserveddatafromthestimulusfeatures
and(2)explorewhichstimulusfeaturesaremostinvolvedinthis
prediction.
2.2. Decoding
| Whereas | encoding | refers | to predicting | the response | given a |     |     |     |     |     |     |     |
| ------- | -------- | ------ | ------------- | ------------ | ------- | --- | --- | --- | --- | --- | --- | --- |
stimulus,decodingreferstopredictingthestimulusfromthemea-
suredresponse,allowingprobingoftherepresentationalcontent
| of neuronal | population |     | recordings (Bialek, | Rieke, de | Ruyter van |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | ------------------- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Steveninck,&Warland,1991;Stanley,Li,&Dan,1999).Decoding
canbeaccomplishedviatheinversionofanencodingmodelsince
| ({x }|{y | })∝p | ({y }|{x | }) ({x }), |     |     |     |     |     |     |     |     |     |
| -------- | ---- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p t      | t    | t        | t p t      |     | (9) |     |     |     |     |     |     |     |
({x })
| wheretheright-handsidefollowsfromBayes’rule.Here,p |     |     |     |     | t is |     |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
astimuluspriorwhichmodelsthestatisticalstructureintheenvi-
ronment.
| If we | assume | that the | system is Markovian | (i.e. | the future is |     |     |     |     |     |     |     |
| ----- | ------ | -------- | ------------------- | ----- | ------------- | --- | --- | --- | --- | --- | --- | --- |
independentofthepastgiventhepresent)thenwemaywrite7
n
| ({x })=p | ( ) | (      | ).  |     |      |     |     |     |     |     |     |     |
| -------- | --- | ------ | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
| p        | x   | p x |x | i−1 |     | (10) |     |     |     |     |     |     |     |
| t        | 1   | i      |     |     |      |     |     |     |     |     |     |     |
i=2 F i g . 2 . D ia g ra m s ho w i ng h ow a s ti m u l u s c a u s e s r e s p o n se s i n m ult i p le p o o l s o f
|     |     |     |     |     |     | n e u r o ns w h | ic h ca n b | e m ea su red n o n | -i n v a si v | e l y . A f e a | t u re s p a | ce φ c h a ra c te r i ze s |
| --- | --- | --- | --- | --- | --- | ---------------- | ----------- | ------------------- | ------------- | --------------- | ------------ | --------------------------- |
t
Normally,theobjectiveistoobtainthemostprobablereconstruc- whatstimulusfeaturesdriveneuronalresponses.Aforwardmodel f(φ)specifies
t
tionofthestimulusgivenobserveddata,which,whencombining how neuronal population activity projects onto the ith sensor. Note that the
Eqs.(6)and(10),isgivenby numberofneuronalpopulationsneednotcorrespondtothenumberofsensors.
Furthermore,eachsensormaypickuptheactivityofmultipleneuronalpopulations
 n  t h a t h a v e t h e i r o w n i n t e r n a l d y n a m i c s ( r e d a r ro w s ) . E n c o d i n g r e f e r s t o p r e d ic t i n g
|     |     |    |     |    |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{x }∗ =argm a x p ( x ) p ( x |x ) p ( y |φ ) . (11) t h e m e a s u r e m e n t s f r o m t h e s t i m u l u s . D e c o d i n g r e f e r s t o p r e d i ct in g t h e s t i m u l u s
t 1 i i−1 t t f r o m t h e m e a s u r e m e n t s a n d c a n b e o b t a in e d b y i n v e r ti n g a n e n c o d i n g m o d e l .
{x t } i=
2 t S p a t i o t e m p o r a l s t a t i s t i c a l d e p e n d e n c i e s i n t h e s e n s o r y i n p u t , a s i l l u s t r a t e d b y
Thisformulationwasusedin,e.g.,vanGerven,Kok,deLange,and the film reel, may affect the neuronal population dynamics in a complex
Heskes (2011), where both the direction and orientation of per- manner. This statistical structure can also be exploited in order to facilitate
stimulusreconstructionduringdecoding(see,e.g.,vanGervenetal.,2011).(For
ceivedrotatinggratingsweredecodedfromfMRImeasurements.
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferred
Incasetherearenotemporaldependencies,eachstimuluscanbe
tothewebversionofthisarticle.)
reconstructedseparatelyforeachresponsevectorataspecifictime
| pointas |     |     |     |     |     | designmatrixwherenisthenumberoftrialsandpthenumberof |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
∗  ( ) ( |φ ). regressors.SPMmakesuseofthegenerallinearmodel(GLM),to
| x =argmax | p   | x p y |     |     | (12) |                             |     |     |     |     |     |     |
| --------- | --- | ----- | --- | --- | ---- | --------------------------- | --- | --- | --- | --- | --- | --- |
| t         |     | t t   | t   |     |      | writetheobservedresponsesas |     |     |     |     |     |     |
xt
Validationofadecodingmodeltypicallyproceedsbycomputing Y=H(cid:56)B+E (13)
| thesimilarity(e.g.correlation)betweenpresentedstimulix |         |     |               |                | and     |                                                 |     |     |     |     |     |     |
| ------------------------------------------------------ | ------- | --- | ------------- | -------------- | ------- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                                                        |         |     |               |                | t       | whereYisthem×qmeasurementmatrix,Hisam×nToeplitz |     |     |     |     |     |     |
| reconstructed                                          | stimuli | x ∗ | on a separate | test set which | has not |                                                 |     |     |     |     |     |     |
t
beenusedduringmodele stimation.Aschematicwhichdepictsthe matrixwhichtakesintoaccountthehemodynamicdelay,Bisthe
×
relationshipbetweenencodinganddecodingisshowninFig.2. p q matrix of regression coefficients and where E is residual
|     |     |     |     |     |     | noise. SPM | uses | the regression | coefficients |     | associated | with a |
| --- | --- | --- | --- | --- | --- | ---------- | ---- | -------------- | ------------ | --- | ---------- | ------ |
smallnumberoftaskregressors(i.e.ptendstobesmall)totest
3. Relatedapproaches
hypothesesaboutregionallyspecificeffects.
OnecouldarguethattheGaussianencodingmodelinEq.(8)is
Toplacetheencodingframeworkwithinabroadercontextof
equivalenttotheformulationofaGLMinEq.(13).Thereishowever
| analysis approaches |     | in cognitive | neuroscience, | a brief | overview |              |            |            |     |         |     |              |
| ------------------- | --- | ------------ | ------------- | ------- | -------- | ------------ | ---------- | ---------- | --- | ------- | --- | ------------ |
|                     |     |              |               |         |          | an important | conceptual | difference |     | between | the | encoding and |
ofalternativepopularapproachesisprovided.Wehereconsider
SPMapproaches.IncontrasttoSPM,encodingmodelsconstitute
| statistical | parametric | mapping, | multivariate | pattern | analysis, |     |     |     |     |     |     |     |
| ----------- | ---------- | -------- | ------------ | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
populationreceptivefieldmappingandrepresentationalsimilarity hypothesesabouthowthebrainprocessesarbitrarysensoryinput,
analysis. ultimatelyleadingtoobservedmeasurements.Inthissense,model
|     |     |     |     |     |     | comparison | becomes | the key     | objective. | To     | function | as veridical  |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | ----------- | ---------- | ------ | -------- | ------------- |
|     |     |     |     |     |     | models of  | sensory | processing, | encoding   | models |          | may use up to |
3.1. Statisticalparametricmapping
|             |            |     |               |               |      | thousands    | of features | to accurately |         | predict | measured   | responses |
| ----------- | ---------- | --- | ------------- | ------------- | ---- | ------------ | ----------- | ------------- | ------- | ------- | ---------- | --------- |
|             |            |     |               |               |      | to arbitrary | sensory     | input (i.e.   | p tends | to      | be large). | Moreover, |
| Statistical | parametric |     | mapping (SPM) | is a commonly | used |              |             |               |         |         |            |           |
approachinneuroimaging(Fristonetal.,1995).Let(cid:56)bethen×p encodingmodelsmayuseforwardmodelsofarbitrarycomplexity
ratherthanbeingrestrictedtotheformulationusedinEq.(8).
3.2. Multivariatepatternanalysis
7 Thisformulationdoesnotprecludemodelingofnon-Markoviandynamicssince
long-termdependenciescanbeeithercapturedusingaugmentedstimulusrepre-
|     |     |     |     |     |     | An alternative |     | analysis approach |     | which | has become | increas- |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ----------------- | --- | ----- | ---------- | -------- |
sentations(capturing,e.g.,stimulusvelocity)orbyencodingthesedependenciesin
inglypopularoverthelastdecadeismultivariate(ormultivoxel)
theneuralpopulationdynamicsitself(cf.Eq.(3)).

176 M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183
patternanalysis(MVPA)(Cox&Savoy,2003;Haxbyetal.,2001; Let Y be the n × q matrix of brain responses and let (cid:56) be the
Haynes & Rees, 2006; Kamitani & Tong, 2005; Miyawaki et al., n × p design matrix. We use U and V to represent the n × n
2008;Norman,Polyn,Detra,&Haxby,2006;Tong&Pratte,2012). dissimilarity matrices constructed from Y and (cid:56), respectively.
Here,thestrategyistodecodestimulionatrial-by-trialbasisby Here,u ij representsthedifference(e.g.oneminusthecorrelation)
computing betweentrialsiandjintermsoftheirrespectivebrainresponses
v
∗ whereas ij represents this difference in terms of the model
| x =argmaxp | ( x | |y ). |     |     |     | (14) |     |     |     |     |     |     |     |
| ---------- | --- | ----- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
t t t predictions.Bycomputingthesimilarity(e.g.correlation)between
xt
|           |          |        |           |                |     |             | (the vectorized | upper | triangular | parts | of) U | and V | one can |
| --------- | -------- | ------ | --------- | -------------- | --- | ----------- | --------------- | ----- | ---------- | ----- | ----- | ----- | ------- |
| Hence, in | contrast | to the | inversion | of an encoding |     | model as in |                 |       |            |       |       |       |         |
obtainevidenceabouthowwellaparticularcomputationalmodel
Eq.(12),MVPAaimstopredictthestimulusdirectlyfromdata.8
explainstheresponsepatternsinaparticularbrainregion.
| In machine | learning | parlance, | encoding | models | are | generative |              |     |     |                    |     |       |           |
| ---------- | -------- | --------- | -------- | ------ | --- | ---------- | ------------ | --- | --- | ------------------ | --- | ----- | --------- |
|            |          |           |          |        |     |            | An advantage | of  | RSA | is that no forward |     | model | is needed |
modelswhereasMVPAmakesuseofdiscriminativemodels,both to compare feature activations with measured responses. This
ofwhichhavetheirownstrengthsandweaknesses(Ng&Jordan, isparticularlycomputationallyadvantageouswhendealingwith
2001). high-dimensional feature spaces. A disadvantage is that RSA
In a neuroscientific context, MVPA has as an advantage that cannotdeterminerepresentation-specificinformationencodedin
region-specific representational content can be probed without individual neuronal responses since similarity matrices must be
requiringanyassumptionsabouthowthatbrainregionresponds
constructedfrommultivariateresponsepatterns.
toastimulus.Conversely,encodingmodelshaveasanadvantage
| that they | make | explicit our | assumptions | on  | how | brain regions |     |     |     |     |     |     |     |
| --------- | ---- | ------------ | ----------- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
4. Constructionoffeaturespaces
respondtostimuli,alsofacilitatingmodelrefinementinorderto
improvemodelfit.
|     |     |     |     |     |     |     | From       | (4), it is clear | that | the notion     | of a   | feature | space is |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ---------------- | ---- | -------------- | ------ | ------- | -------- |
|     |     |     |     |     |     |     | central to | the development  |      | of an encoding | model. | The     | question |
3.3. Populationreceptivefieldmapping becomeswhatconstitutesasuitablefeaturespace.Tosimplifythe
exposition,weassumethattheencodingmodeliswritteninthe
AnotherrelatedapproachispRFmapping(Wandell&Winawer,
linearizingformofEq.(8).Furthermore,withoutlossofgenerality,
2011),wherethegoalistomappRFparametersestimatedfrom
weassumethattheresponseisinstantaneous.Thatis,wewritethe
fMRIdatabackontocortexinordertorevealthefunctionalorga-
expectedresponseacrossneuronalpopulationsas
nizationofsensorycortices.ConsideragainEq.(2).InpRFmapping,
|                                                        |     |     |     |     |     |             | µ =BTφ                                                   | .   |     |     |     |     | (16) |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | ----------- | -------------------------------------------------------- | --- | --- | --- | --- | --- | ---- |
| itisassumedthatthepopulationreceptivefieldw            |     |     |     |     | i   | ofeachvoxel | t                                                        | t   |     |     |     |     |      |
| icanbegeneratedusingasmallnumberofvoxel-specificparam- |     |     |     |     |     |             |                                                          |     |     |     |     |     | φ    |
|                                                        |     |     |     |     |     |             | Intheliterature,researchershavemadedifferentproposalsfor |     |     |     |     |     | t ,  |
etersthatneedtobeestimatedfromdata.Theseparametersare
asdescribedinthefollowing.
crucialinpRFmappingastheyaremappedbackontocortex.
| For example,         |     | in a Gaussian |             | pRF model,          | for a | voxel i, these |                           |     |     |     |     |     |     |
| -------------------- | --- | ------------- | ----------- | ------------------- | ----- | -------------- | ------------------------- | --- | --- | --- | --- | --- | --- |
|                      |     |               | µ           | σ2ofaGaussiankernel |       |                | 4.1. Linearencodingmodels |     |     |     |     |     |     |
| parametersarethemean |     |               | andvariance |                     |       |                |                           |     |     |     |     |     |     |
|                      |     |               | i           | i                   |       |                |                           |     |     |     |     |     |     |
 ∥µ − ( , )T∥2 A str aig h t for w a rd a pp r o a c h i s t o a s su m e t h a t φ = x . T hi s
|     |     |     | x y |     |     |     |     |     |     |     |     | t   | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
k ( x , y )=exp − i (15) cho ice im p le m e nt s th e n o t io n t h a t t h e m e as u r e d r es pon s e is a
| i   |     | σ   | 2   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2
|     |     |     | i   |       |     |     | linearcombinationofthesensoryinputs.Whilethislinearityas- |     |     |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | ( , ) |     |     |                                                           |     |     |     |     |     |     |
which defines, for each location x y in visual space, the sumptionisquiterestrictive,suchamodelhasbeenshowntoout-
responsivenessofvoxelitothatlocation.Ifweassociateeachinput performstandardapproachesforpRFmapping(Lee,Papanikolaou,
( , )
withapoint x y insensoryspacethenwecanconstructthenth Logothetis,Smirnakis,&Keliris,2013).Furthermore,incontrastto
n n
elementoftheithGaussianfilteras ( w ) = k ( x , y ) .Itisthese more complex models, it has as an advantage that the decoding
|     |     |     |     | i n | i n | n   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parameters,estimatedpervoxel,thatallowmappingofreceptive modelfollowsanalyticallyfromtheencodingmodel.LetRdenote
fieldsizes σ2andlocations µ2acrossthecorticalsurface,thereby thepixelcovarianceestimatedfromthe(demeaned)trainingdata.
|     | i   |     | i   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
revealingth eretinotopicorga nizationofvisualcortex. GivenregressioncoefficientsB,noisecovariance(cid:54)andastimulus
Summarizing, pRF mapping can be seen as a special kind of priorx ∼ N( 0 , R ) ,theoptimalreconstructioncanbeshownto
equal
| parametric | encoding | model | whose | feature | space | is determined |     |     |     |     |     |     |     |
| ---------- | -------- | ----- | ----- | ------- | ----- | ------------- | --- | --- | --- | --- | --- | --- | --- |
by a small number of parameters of interest that can be easily = −1
|                                                          |     |     |     |     |     |     | x ∗ R −1+B(cid:54)−1BT |     | B(cid:54)−1y | .   |     |     | (17) |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | ------------ | --- | --- | --- | ---- |
| mappedbackontothecorticalsurface.ExtensionsoftheGaussian |     |     |     |     |     |     | t                      |     |              | t   |     |     |      |
pRFmodelaimtobetteraccountforunexplainedvarianceusing
Usingthisapproach,asillustratedinFig.3(A),perceivedimages
| richer parameterized |     | distributions, |     | e.g. by | modeling | surround |                |            |     |                       |     |               |     |
| -------------------- | --- | -------------- | --- | ------- | -------- | -------- | -------------- | ---------- | --- | --------------------- | --- | ------------- | --- |
|                      |     |                |     |         |          |          | of handwritten | characters |     | could be successfully |     | reconstructed |     |
suppressionusingadifferenceofGaussians(Zuiderbaan,Harvey,
|     |     |     |     |     |     |     | from measured | fMRI | response | patterns | (Schoenmakers |     | et al., |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | -------- | -------- | ------------- | --- | ------- |
&Dumoulin,2012).
2013).ExtensionsthatmakeuseofaGaussianmixturepriorand
semanticgatingfurtherimproveonthisapproach(Schoenmakers,
3.4. Representationalsimilarityanalysis Güçlü,vanGerven,&Heskes,2015).Still,thelinearityassumption
usedinconjunctionwithaGaussianpriordoesnotgeneralizewell
Yetanotherapproachwhichhasgainedtractioninrecentyears tomorecomplexnaturalisticdatasets.
| is representational |        | similarity | analysis | (RSA)              | (Kriegeskorte, | Mur,        |     |     |     |     |     |     |     |
| ------------------- | ------ | ---------- | -------- | ------------------ | -------------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| & Bandettini,       | 2008). | One        | goal of  | RSA is to evaluate |                | whether the |     |     |     |     |     |     |     |
4.2. Nonlinearencodingmodels
| representational |     | structure | of a | particular computational |     | model |     |     |     |     |     |     |     |
| ---------------- | --- | --------- | ---- | ------------------------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
matchesthatoftheresponsepatternsinaparticularbrainregion.
Amoresophisticatedapproachistouseafeaturespaceofthe
form
|     |     |     |     |     |     |     | φ =h ( WTx | ),  |     |     |     |     | (18) |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | ---- |
|     |     |     |     |     |     |     | t          | t   |     |     |     |     |      |
8 SeevanBergen,Ma,Pratte,andJehee(2015)forarecentexamplewherethe
wheretheneuralresponsedependsonlinearfilters(columnsofW)
sensoryuncertaintyratherthansensoryrepresentationwasdecodedusinganMVPA
|     |     |     |     |     |     |     | thatareappliedtothestimulus, |     |     | followedbyastaticnonlinearity |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | ----------------------------- | --- | --- | --- |
approach.

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 177
Fig.3. Examplesofresultsobtainedusingencodingmodels.(A)AnexampleofhandwrittencharacterstimuliandtheirreconstructionsbasedonfMRIresponsepatterns
usingthelinearmodel(Schoenmakersetal.,2015).(B)ReconstructionofimaginedpatternsfromhumanbrainactivityusingGaussianfilters.Blackcirclesdenotethe
imaginedpatternsinleft(LH)andright(RH)hemifieldsandheatmapsshowtheirreconstructions.ReprintedfromThirionetal.(2006)withpermissionfromElsevier.
(C)ExamplesofaGaborwavelet(GW)andasparsecoding(SC)filter.Red(blue)valuesindicateareasinvisualspacethatactivate(suppress)thefilters.(D)Assignments
ofvoxelstothosedeepneuralnetworkslayersthatbestexplainvoxelresponsesrevealahierarchicalorganizationofneuralrepresentationsacrosstheventralstream.(For
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
Source:ReprintedfromGüçlüandvanGerven(2015).
(·)
h (Kay,Winawer,Mezer,&Wandell,2013).Thisisverycloseto hasbeenshownthatbrainresponsestonaturalisticimagescould
thedefinitionofthepRFmodelofEq.(2).However,inthiscase, bepredictedwithhighaccuracy.TheGaussianfilterhasalsobeen
a vector of putative pRF model responses is obtained which are used by Thirion et al. (2006) to reconstruct perceived and even
thenlinearlymappedontotheobservedresponsesviathematrixB. imaginedpatternsfrommultivariatefMRIresponsepatterns(see
(·)
| Notethatthenonlinearityh |     |     | isessentialsince,withoutit,using |     |     |     |     | Fig.3(B)). |     |     |     |     |     |     |
| ------------------------ | --- | --- | -------------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
Eq.(16),wemaywrite µ = BTφ = CTx withC = WB,which Asimilarmodelhasbeenusedtopredictpopulationresponses
|     |     |     |     | t   | t   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
isnomoreexpressivethanthelinearformulation.Moreover,the to dynamically changing visual stimuli. Nishimoto et al. (2011)
nonlinearity allows for the implementation of various nonlinear used a nonlinear motion–energy model based on spatiotempo-
neuralresponsepropertiesthatareknowntohold(Kay,Winawer, ral Gabor filters to accurately predict brain responses to natu-
ralisticmoviesandevenreconstructmoviesfromobservedbrain
Rokem,Mezer,&Wandell,2013;Wandell,Winawer,&Kay,2015).
responses.Thisisimpressivegiventhehighsamplingrateofnat-
Asanexample,considerthenonlinearGaborwaveletpyramid
|     |     |     |     |     |     |     |     | uralistic | movies compared | to  | the low | sampling | rate | of fMRI re- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------------- | --- | ------- | -------- | ---- | ----------- |
(GWP)model(Kayetal.,2008;Naselarisetal.,2009)whichuses
| localcontrastenergyfeaturescomputedfrom2DGaborwavelets. |     |     |     |     |     |     |     | sponses. |     |     |     |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
Gaborwaveletshavebeenusedtomodelthepropertiesofneurons While nonlinear encoding models improve encoding perfor-
in early visual cortex (Daugman, 1985; Jones & Palmer, 1987; mance over linear ones, they do prohibit decoding via analyti-
Marčelja,1980).AGaborwaveletwithunitgainandzerooffsetcan cal inversion as in Eq. (17) since the used nonlinearity may be
bewrittenas non-invertible. Instead, decoding can be achieved via numerical
|     |     |     |      |     |     |     |     | inversion. | For example, | one | can make | use of | an empirical | prior |
| --- | --- | --- | ---- | --- | --- | --- | --- | ---------- | ------------ | --- | -------- | ------ | ------------ | ----- |
|     |    | ω2  | ρ2  |     |     |     |     |            |              |     |          |        |              |       |
( , )=exp cos(2 π ρ +φ) (Naselaris et al., 2009). This prior assumes that the stimuli are
| k x y | −   | i − | i   |     | f   |     | (19) |     |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
i 2 σ2 2 ν2 i i i completelycharacterizedbyalargebutfinitesetXofcandidate
|     |         | i   | i      |     |       |      |       |              |                       |               |             | ( )     |       |                |
| --- | ------- | --- | ------ | --- | ----- | ---- | ----- | ------------ | --------------------- | ------------- | ----------- | ------- | ----- | -------------- |
|     |         |     |        |     |       |      |       | s ti m u l i | o f e q u a l p r o b | a b ili t y , | s u c h t h | a t p x | ∝ 1 ∈ | w h e r e 1 is |
| ω   | =( x−x¯ | ) θ | +( −y¯ | ) θ | ρ =−( | x−x¯ | ) θ + |              |                       |               |             |         | x     | X              |
with i i cos i y i sin i and i i sin i t h e i n d ic a t o r fu n c t i o n . In t h i s c a s e, t h e o p t im a lre c o n str u c ti o n be -
| ( − y¯ | ) θ |     | ( ,σ 2,ν | 2) φ | ( x¯ , y¯ ) | θ   |     |     |     |     |     |     |     |     |
| ------ | --- | --- | -------- | ---- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
y i cos i . Here, f i , i , i i and i determine comesx ∗ = argmax p ( y | x ) ,whichcanbecomputedbyit-
|     |     |     | i i |     |     |     |     |     | x∈X |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the spatial scale, phase, location and orientation of the wavelet, eratingoverallstimuliinthesetandcomputingitslikelihood.If
respectively.9 Analogous to the Gaussian filter, we construct the thetargetstimulusispartofthecandidatesetthenwerefertothis
|     |     |     |     | (   | )   | ( , ) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
nthelementoftheithGaborfilteras w = k x y .Byusing procedureasidentificationratherthanreconstruction.
|               |                                                    |      |               |     | i n          | i n n |           |                                    |     |         |                    |     |     |               |
| ------------- | -------------------------------------------------- | ---- | ------------- | --- | ------------ | ----- | --------- | ---------------------------------- | --- | ------- | ------------------ | --- | --- | ------------- |
| these filters | together                                           | with | a compressive |     | nonlinearity |       | h ( x ) = |                                    |     |         |                    |     |     |               |
| log (|x|+1    | ) asthefeaturespaceforalinearizingencodingmodel,it |      |               |     |              |       |           |                                    |     |         |                    |     |     |               |
|               |                                                    |      |               |     |              |       |           | 4.3. Learningfeaturespacesfromdata |     |         |                    |     |     |               |
|               |                                                    |      |               |     |              |       |           | An alternative                     | to  | the use | of a predetermined |     |     | feature space |
whichembodiesourassumptionsaboutstimulus–responserela-
9 SeeVu,Ravikumar,andNaselaris(2011)foranalternativeformulationofthe
tionshipsistouseamoreobjectiveapproach,wherethefeature
Gaborwaveletintermsofaproductofacomplex2DFourierbasisfunctionanda
spaceislearntdirectlyfromdata.Theunderlyingassumptionhere
Gaussiankernel.

178 M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183
isthatthebrainisoptimizedtoextractstatisticalregularitiesfrom representations across the cortical sheet. One way to tackle this
theenvironment(Blakemore&Cooper,1970;Marr,1970;Purves, problemistousehierarchicalmodelsconsistingofmultiplelayers
| Wojtach, & | Lotto, | 2011; Rao | &   | Ballard, | 1999; | Simoncelli | & Ol- |     |     |     |     |     |     |     |
| ---------- | ------ | --------- | --- | -------- | ----- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- |
ofelementaryunitsthataretrainedtoprocessincreasinglycom-
shausen,2001).Thissuggeststhatastatisticalmodelthatlearns plexfeaturesoftheirenvironment.
invariancesfrommassivedatasetsmayconstituteanoptimalfea- Recently,hierarchicalmodelsofthissort,knownasdeepneural
turespace.InGüçlüandvanGerven(2014)asparsecoding(SC) networks(DNNs),havebeenusedtopredictneuralresponsesin
modelwasusedwhichlearnssimple-andcomplex-cellresponses downstream cortical areas. From a neuroscientific point of view
inanunsupervisedmannerbypresentingthemodelwithalarge DNNs are of interest since their ability to detect increasingly
numberofnaturalimagepatches(Hyvärinen&Hoyer,2001).Inter- abstractpropertiesoftheirenvironmentisreminiscentoftheway
estingly,itwasshownthatresponsespredictedbytheSCmodel sensoryinputisprocessedacrossdifferentsensorypathways(Cox
significantly outperformed those of the GWP model (Kay et al., & Dean, 2014; Fukushima, 1980; Riesenhuber & Poggio, 1999;
2008).ThisimpliesthattheGWPmodelremainswantingwhenit Serre,Wolf,Bileschi,Riesenhuber,&Poggio,2007).ADNNlearns
comestoexplainingresponsesinearlyvisualcortex.SeeFig.3(C)
|                |     |         |         |     |          |        |         | a feature space | in  | a supervised | manner | and | has been shown | to  |
| -------------- | --- | ------- | ------- | --- | -------- | ------ | ------- | --------------- | --- | ------------ | ------ | --- | -------------- | --- |
| for an example | of  | a Gabor | wavelet | and | a sparse | coding | filter, |                 |     |              |        |     |                |     |
provideexcellentpredictionsofneuralresponses.Briefly,givena
showingbothdifferencesandsimilarities.Animportantupshotof = {( , ),...,( , )},consisting,
|     |     |     |     |     |     |     |     | labeledtrainingdatasetZ |     |     | x 1 z | 1   | x N z N |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ----- | --- | ------- | --- |
this work is that it is possible to validate computational models for example, of images x j and their categories z, j a hierarchy
φ
oflearningandinferenceinsensorycortexusingnon-invasively of K non-linear transformations is learned using (stochastic)
k
| recordedneuraldata.                 |     |     |     |     |     |     |     | optimizationsuchthat |     |     |     |     |     |      |
| ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | ---- |
|                                     |     |     |     |     |     |     |     | zˆ =(φ ◦···◦φ        | )(  | )   |     |     |     |      |
| 4.4. Probingsemanticrepresentations |     |     |     |     |     |     |     |                      |     | x i |     |     |     | (20) |
|                                     |     |     |     |     |     |     |     | K                    | 1   |     |     |     |     |      |
isoptimalinthesensethatanobjectivefunctionquantifyingthe
WhiletheGWPmodelandrelatedmodelssuchastheSCmodel
|                                                          |     |     |     |     |     |     |     | distancebetweenzˆandz |     | isminimized(LeCun,Bengio,&Hinton, |     |     |     |     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --------------------------------- | --- | --- | --- | --- |
| capturelow-levelstimulusfeaturesthatinformaboutresponses |     |     |     |     |     |     |     |                       |     | i                                 |     |     |     |     |
2015).Subsequently,anysubsetofthefeaturesembeddedinthe
| in early visual | cortex, | they | are | unable | to capture | responses | to  |     |     |     |     |     |     |     |
| --------------- | ------- | ---- | --- | ------ | ---------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
high-levelstimulusfeaturesthatinformaboutsemanticrepresen- hierarchical model can be used as an encoding model’s feature
tationsindownstreamassociationareas. spaceasdefinedinEq.(4),whichisthenmappedontotheobserved
responses(cf.Eq.(5)).
Onewaytocircumventthisproblemistomanuallyconstructa
|     |     |     |     |     |     |     |     | Researchers | have | begun | to use | DNNs to | predict neural | re- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ----- | ------ | ------- | -------------- | --- |
featurespace,asintheGWPmodelforearlyvisualcortex,butnow
|     |     |     |     |     |     |     |     | sponsesto | complexnaturalistic |     | stimuli,yielding |     | state-of-the-art |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------------- | --- | ---------------- | --- | ---------------- | --- |
gearedtowardsthedirectrepresentationofsemanticinformation.
|               |     |       |              |     |      |               |     | results (Kriegeskorte, |     | 2015). | Khaligh-Razavi |     | and Kriegeskorte |     |
| ------------- | --- | ----- | ------------ | --- | ---- | ------------- | --- | ---------------------- | --- | ------ | -------------- | --- | ---------------- | --- |
| This approach | was | used, | for example, |     | in a | seminal study | by  |                        |     |        |                |     |                  |     |
(2014)usedDNNsinconjunctionwithRSAtoexplainresponsesin
Mitchelletal.(2008),wherethefeaturespaceforwrittennouns
wasmodeledintermsoftheirassociationstrengthswithdifferent ITcortex.TheirresultswereconsistentwiththoseofYaminsetal.
verbs. It was shown that this model could predict responses to (2014)whohaveshownthatresponsesinMonkeyITarebestpre-
nouns that were not previously seen by the encoding model by dictedbyhierarchicalmodelsthatareexplicitlyoptimizedforob-
using the mapping from nouns to verbs in conjunction with the jectcategorization.
mappingfromverbstofMRIresponses. Güçlü and van Gerven (2015, in press) have subsequently
InNaselarisetal.(2009),alow-levelstructuralencodingmodel, shownthatDNNlayershigherupinthehierarchybetterencode
predicting responses in early visual cortex, was combined with the responses of brain regions as one moves downstream along
boththeventralanddorsalvisualpathways(seeFig.3(D)).Based
| a semantic | encoding | model, | predicting |     | responses | to stimulus |     |     |     |     |     |     |     |     |
| ---------- | -------- | ------ | ---------- | --- | --------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
category in downstream areas. The integration of both sources on MEG data, Cichy, Khosla, Pantazis, Torralba, and Oliva (2016)
ofinformationtogetherwiththeuseofanempiricalpriorledto used an RSA approach to show that this gradient is not only
high-qualityreconstructions. reflected in space, with deeper DNN layers corresponding best
Huth,Nishimoto,Vu,andGallant(2012)annotatedthousands withdownstreambrainregions,butalsointime,withdeeperDNN
ofmovieframeswithhundredsofobjectandactioncategoriesto layersbeingmostlyexpressedatlaterpointsintime.
createafeaturespacewithwhichtoexaminetherepresentationof Whiledeepneuralnetworkshavemainlybeenappliedinthe
semanticknowledgeacrossthehumanbrain.Principalcomponent context of visual perception, they can equally well be applied
analysisoftheregressioncoefficientsthatmapthefeaturespace
|     |     |     |     |     |     |     |     | to auditory | data in | order to | understand | population | responses | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | -------- | ---------- | ---------- | --------- | --- |
tovoxelresponsesrevealedanumberofdimensionsthathadan
auditorycortex.SeeGüçlü,Thielen,Hanke,andvanGerven(2016)
intuitivesemanticinterpretationintermsofanimacy,socialcat-
andKelletal.(2015)forinitialworkinthisdirection.
egoriesandman-madeversusnaturalstimuli.Recently,asimilar Alongthesamelines,neuralnetworkscanbemadesuitablefor
approachhasbeenusedbythesameauthorstomapsemanticin- processingoflinguisticinputusingwordembeddingslearnedfrom
formationacrossthecorticalsheetusingfMRIdataobtainedwhile largelinguisticcorporabasedontheskip-grammodel(Mikolov,
subjectswerelisteningtoautobiographicalstories(Huth,deHeer, Sutskever, Chen, Corrado, & Dean, 2013). See Güçlü and van
Griffiths,&Gallant,2016). Gerven (2016) and Nishida, Huth, Gallant and Nishimoto (2015)
forneuroscientificapplicationsofthismodel.
4.5. Constructinghierarchicalfeaturespaces
Finally,HorikawaandKamitani(2016)haverecentlydemon-
stratedthatDNNsevenallowthedecodingofimaginedobjectcat-
Alimitationofthemanualannotationofstimuliintermsofse- egoriesfromfMRImeasurements.
manticpropertiesisthatitislaboriousandmayintroduceanno-
tatorbias.Moreimportantly,itdoesnotinformabouthowthese
|     |     |     |     |     |     |     |     | 5. Applicationofencodingmodels |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- |
semanticrepresentationscomeaboutinactualbrains.Inasense,
human-definedfeaturespacescanbeviewedascomputationalor-
aclesthatdonotprovideanexplicitaccountofneuralprocessing The previous section reviewed state-of-the-art approaches in
ofsensorystimuli.Ideally,wewouldliketomodelhowlow-,mid- the development of increasingly sophisticated feature spaces.
andhigh-levelstimulusfeaturesemergebycommittingtoalearn- Here,weexplorehowencodingmodelsareusedtoprovidenovel
ingprocedure.Thesefeaturescansubsequentlybeusedtoprobe insightsinsensoryneuroscience.

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 179
5.1. Revealingthefunctionalorganizationofsensoryrepresentations ofotherobjectivefunctions,liketheoneusedindeepreinforce-
mentlearning(Mnihetal.,2015).Thatis,whilemaximizingapar-
Encodingmodelsallowforthecharacterizationofhowparticu- ticular objective function can be sufficient for explaining neural
larstimuluspropertiesaremappedacross(sub)corticalareas,pro- responses,itmightnotbenecessary.Inotherwords,onecannot
viding a macroscopic view of the neural organization of sensory directlyinferfunctionfromreceptivefieldproperties(Churchland
representations at the level of neuronal populations (DeSimone, &Sejnowski,1990).
Viviano, & Schneider, 2015; Sanger, 2003; Wandell et al., 2015).
Thiscan,forexample,beachievedbyidentifyingwhichstimulus 5.3. Investigatingmodulationsofpopulationresponses
featuresdriveavoxelibyinspectingthevectorofregressionco-
β
efficients i .However,thischaracterizationistypicallybasedon Sofar,encodingmodelshavebeendescribedintermsoffeed-
theuseofahigh-dimensionalfeaturespace,whichcomplicatesthe forwardprocessingofsensoryinformation.However,thebrainis
analysis.Thatis,giventhelargenumberoffeatures,significance notapassiveinputdevice.Top-downchangesinattention(Çukur
testingonindividualmodelweightsasinanSPManalysisnotonly etal.,2013),expectation(Summerfield&deLange,2014)andtask
becomescumbersome,butalsoproblematicduetomulticollinear- demand(Fritz,Shamma,Elhilali,&Klein,2003;McKeeetal.,2014)
ity(Monti,2011). induce population receptive field reconfigurations (Gilbert & Li,
To resolve this issue, different approaches have been used to 2013),whichcanbeprobedusingencodingmodels.Forexample,
project the data onto a lower-dimensional space. For example, attentionhasbeenshowntoincreasepRFgain,eccentricity,and
Huth et al. (2012) used a principal components analysis on sizeinvisualcortex(Kay,Weiner,&Grill-Spector,2015;Klein,Har-
modelweightstorevealhowsemanticinformationisrepresented vey,&Dumoulin,2014).BrouwerandHeeger(2013)usedapRF
acrosscortex.GüçlüandvanGerven(2015)insteadcharacterized modeltoshowthattherepresentationofcolorindownstreamvi-
stimulus features in terms of their Kolmogorov complexity in sualareaschangedtoamorecategoricalrepresentationinacolor
ordertoshowhowincreasinglycomplexstimuluspropertiesare namingtaskcomparedtoadivertedattentiontask.
encodedacrosstheventralstream. Changesinpopulationresponsesmayalsobeinducedby(re-
Alternatively,onemayresorttotheuseofwhatwerefertoas versible)learningeffects(Rainer,Lee,&Logothetis,2004),develop-
simulatedneurophysiology,whereencodingmodelsareusedpost mentalchange(Johnson&Munakata,2005)orneurodegeneration
hoc to predict how neural populations respond to basic stimuli (Brewer&Barton,2014).Encodingmodelscaninprinciplebeused
thataremoreeasilycharacterizable(Güçlü&vanGerven,2014; toprobethesefleetingorlong-lastingchangesintheorganization
Kay et al., 2008; Kay, Winawer, Rokem et al., 2013; Nishimoto oflow-,mid-andhigh-levelrepresentationsacrossthecorticalsur-
etal.,2011).Forexample,GüçlüandvanGerven(2014)usedpoint face,forexample,incaseofneurodegeneration,bycomparingthe
stimuliandsinewavegratingsasinputtoanencodingmodelto predictionsofneuralresponsepatternsinpatientsandcontrols.
estimateRFlocation,orientationandspatialfrequencytuningof Also within the general population, variability in population
individualvoxels. responses across cortex is to be expected. These differences can
Yet another approach is to use each feature individually to beexploredbydevelopingencodingmodelstailoredtoindividual
predict BOLD responses and to analyze the resulting response subjects. This constitutes a shift in research focus from finding
patterns.Forexample,clusteranalysisofresponsepatternsinarea commonalities between subjects using group-level analyses,
LO hinted at subsets of voxels that were sensitive to different towards understanding the detailed functional organization of
groupsofstimulusfeatures(Güçlü&vanGerven,2015).Thismore neural representations in individual subjects as driven by more
detailedmappingofrepresentationsacrosscortexisparticularly abstractkeycomputationalprinciples(Stelzer,Lohmann,Mueller,
important for unraveling the elusive characteristics of mid- to Buschmann,&Turner,2014).
high-level neural representations (Carandini et al., 2005; Orban,
2008;Peirce,2015). 5.4. Peeringintothehumanmind
5.2. Encodingmodelsashypothesesaboutbrainfunction Whereasencodingmakesexplicitourassumptionsabouthow
stimuli are represented in the brain, decoding allows us to peer
An important utility of the encoding approach lies in testing intothehumanmindbyreconstructingstimulifrommultivariate
different computational models against each other. Basically, response patterns (Tong & Pratte, 2012). This allows one to
each estimated encoding model implements a particular theory address,forexample,whatstimulusinformationispresentwithin
concerninghowsensoryinformationisrepresentedinthebrain. particularregionsofinterest.Italsoinformsaboutthosestimulus
By comparing encoding models in terms of how well they propertiesthatarenotaccuratelycapturedbytheusedencoding
explainobserveddata,wemayselectamongcompetingtheories. models,asthiswouldrevealitselfintermsofdegradedstimulus
This makes it possible to test whether proposed mechanisms reconstructions.
underlyingtheemergenceofneuralrepresentationsindeedlead Decoding also affords the reconstruction of percepts that are
tomoreprecisepredictionsofneuralresponsestosensoryinput. notadirectconsequenceofexternalbottom-upinputbutrather
For example, the observation that sparse coding models out- oftop-downinternaldrive.Onemaythinkhereofrepresentations
performGaborwaveletmodels(Güçlü&vanGerven,2014)lends induced by imagery (Horikawa & Kamitani, 2016; Naselaris,
support to the hypothesis that sparse coding could be a driving Olman,Stansbury,Ugurbil,&Gallant,2015;Parkkonen,Andersson,
principlebehindtheorganizationofreceptivefieldsinstriatecor- Hämäläinen,&Hari,2008;Stokes,Thompson,Cusack,&Duncan,
tex(Olshausen&Field,1996).Likewise,recentobservationsthat 2010;Thirionetal.,2006;Vetter,Smith,&Muckli,2014),selective
hierarchical models which are optimized for object categoriza- attention(Kamitani&Tong,2005;Niazietal.,2014),perceptual
tionprovidestate-of-the-artpredictionsacrossdifferentareasof filling-in(Chong,Familiar,&Shim,2015),visualillusions(Corney
theventralstream(Güçlü&vanGerven,2015;Khaligh-Razavi& &Lotto,2007;Kok&deLange,2014),memoryprocessing(Harrison
Kriegeskorte,2014;Yaminsetal.,2014)suggestthattheventral & Tong, 2009; Polanía, Paulus, & Nitsche, 2012; Serences, Ester,
streamisexplicitlyoptimizedforobjectcategorization. Vogel,&Awh,2009)anddreaming(Horikawa,Tamaki,Miyawaki,
Atthesametime,oneshouldbecarefulindrawinginferencesof &Kamitani,2013).Decodedrepresentationsmayalsobedirectly
thissort.Thestimulusfeaturesthatemergebyexplicitlyoptimiz- related to overt behavior (Haynes et al., 2007; Williams, Dang,
ingforobjectcategorizationmayalsoemergeviatheoptimization & Kanwisher, 2007). Finally, if internal representations can be

180 M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183
directlydecodedfromhumanbrainactivity,thenthismayspurthe Inpractice,thismappingisextraordinarilycomplex,involvingin-
developmentofnewkindsofbrain–computerinterfacesthatare tricateshort-andlong-rangeinteractionsbetweenneuronalpop-
drivenbychangesininternallyinducedmentalstates(vanGerven ulations in space and time that can only be partially observed
etal.,2009). throughacquiredmeasurements.Toimprovethesensitivityofen-
|     |     |     |     |     |     |     |     | coding models, |     | it is imperative |     | that more | realistic | assumptions |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ---------------- | --- | --------- | --------- | ----------- | --- |
6. Futuredevelopments aboutneuralinformationprocessingareincorporated.
|     |     |     |     |     |     |     |     | Already | at the | sensory | epithelia, |     | the incoming | stimulus | is  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ------- | ---------- | --- | ------------ | -------- | --- |
In the previous section, possible applications of encoding transducedincomplexways(Gollisch&Meister,2010;Masland,
modelshavebeensketched.Here,anumberofresearchdirections 2012).Furthermore,processingofthesestimuliisanactiveprocess
arepointedoutthatareexpectedtofurtheradvancecomputational where sensory input is selected such as to maximize reward or
modelinginsensoryneuroscience. informationgain(Ahmad&Yu,2013).Neuralpopulationactivity
|     |     |     |     |     |     |     |     | itself is characterized |     | by  | intricate | dynamics |     | and may | not only |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --------- | -------- | --- | ------- | -------- |
encoderepresentationsofinvariantsinourenvironmentbutalso
6.1. High-resolutionmappingofneuralrepresentations
|     |     |     |     |     |     |     |     | their sensory | uncertainty |     | (van | Bergen | et al., 2015). | By  | modeling |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ----------- | --- | ---- | ------ | -------------- | --- | -------- |
So far, encoding models have mainly been applied to fMRI such processes, more realistic encoding models can be obtained
datacollectedusingrelativelystandardacquisitionprotocolswith (alsoseeEdelman,2015).
repetition times on the order of 1 s and voxel sizes of about Concerning the employed forward models it should be men-
2 millimeter isotropic. In the near future we can expect to see tioned that encoding models can in principle use arbitrarily
more examples of encoding models applied to high spatial and complex forward models, replacing standard linear regression
temporalresolutionfMRIdataacquiredundernovelprotocolsat methods.Inneuroimaging,thesemodelscanprovidemoreexplicit
implementationsofthetransformationfromneuralresponsesto
ultra-highfieldstrengths(Barth&Poser,2011).Atsomepoint,such
theBOLDresponse,asinducedbyneurovascularcoupling(Aquino,
studiesmayallowprobingofrepresentationalcontentatthelevel
ofindividualcorticalcolumns(Yacoub,Harel,&Ugurbil,2008)as Robinson,&Drysdale,2014;Kay,Winawer,Rokemetal.,2013).In
wellascorticallayers(Kok,Bains,vanMourik,Norris,&deLange, electrophysiology,thesemodelscanincorporateexplicitmecha-
2016;Mucklietal.,2015).Researchwillalsobefacilitatedbyrecent nismsforspikegeneration(Paninski,2004).
initiativesthatprovideopen-accesshigh-resolutionfMRIdatasets
thataretailoredtoencodingstudies(Hankeetal.,2014). 6.3. Towardsbiologicallyplausibleneuralnetworkmodels
| As described |     | earlier on, | encoding | models | can | also be | used in |     |     |     |     |     |     |     |     |
| ------------ | --- | ----------- | -------- | ------ | --- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
conjunctionwithotherimagingmodalities.Moredetailedinsights
Whenconsideringthestate-of-the-artfeaturespacesafforded
| concerning | the spatio-temporal |     | dynamics |     | of neural | representa- |     |         |            |        |          |     |              |     |            |
| ---------- | ------------------- | --- | -------- | --- | --------- | ----------- | --- | ------- | ---------- | ------ | -------- | --- | ------------ | --- | ---------- |
|            |                     |     |          |     |           |             |     | by deep | artificial | neural | networks | it  | is important |     | to realize |
tionscanbegainedusingnon-invasiveEEG/MEGrecordingtech-
|     |     |     |     |     |     |     |     | that while | they | replicate | some | of the | response | properties | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | --------- | ---- | ------ | -------- | ---------- | --- |
niques(Cichyetal.,2016;Clarke,Devereux,Randall,&Tyler,2014;
|     |     |     |     |     |     |     |     | neurons | across | thesensory | stream, | they | make | assumptions | that |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ---------- | ------- | ---- | ---- | ----------- | ---- |
Pantazis,Oliva,&Cichy,2014;Scholte,Ghebreab,Waldorp,Smeul-
areunrealisticfromaneurobiologicalpointofview,bothregarding
ders,&Lamme,2009;vandeNieuwenhuijzenetal.,2013).Theuse the learning procedure with which neural networks are trained
ofmoreinvasiverecordingtechniques(Kobatake&Tanaka,1994; as well as their assumed architecture (Robinson & Rolls, 2015).
QuianQuiroga,Reddy,Kreiman,Koch,&Fried,2005;Yaminsetal., Forexample,whileDNNstrainedinasupervisedmannerprovide
2014)mayleadtoanewunderstandingofthereceptivefieldsof
|     |     |     |     |     |     |     |     | state-of-the-art |     | performance | on  | object | categorization |     | tasks, the |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----------- | --- | ------ | -------------- | --- | ---------- |
singleneuronsorsmallneuronalpopulations.Inpractice,theuse
|     |     |     |     |     |     |     |     | employed | learning | scheme | is  | far removed |     | from learning | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------ | --- | ----------- | --- | ------------- | --- |
ofothermodalitiesonlyrequiresthedevelopmentofappropriate theirbiologicalcounterparts(Bliss&Lømo,1973).10Furthermore,
forwardmodelsthatlinkfeaturespacestoobservedresponses.
|     |     |     |     |     |     |     |     | cortical processing |     | is determined |     | not | only by | feedforward | input |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ------------- | --- | --- | ------- | ----------- | ----- |
Sinceencodingmodelsrequiremassiveamountsofdatainor-
|     |     |     |     |     |     |     |     | but also | by lateral | and | feedback | connections |     | (cf. Eq. | (3)), both |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | -------- | ----------- | --- | -------- | ---------- |
dertoaffordpropermodelestimation,itisimportanttodevelop
withinandbetweencorticalandsubcorticalneuronalpopulations
techniquesthatmakeoptimaluseofthedataavailable.Oneway
(Mumford,1992).
toachievethisistoimposeadditionalrealisticconstraintsonthe
Asatentativeexampleofaneuralnetworkarchitecturewhich
models(seealsoSection6.2).Also,betterremovalof(physiolog-
incorporatestop-downprocessingconsidertheworkpresentedin
| ical) noise | (Kay, | Rokem, Winawer, |     | Dougherty, | &   | Wandell, | 2013; |     |     |     |     |     |     |     |     |
| ----------- | ----- | --------------- | --- | ---------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
vanGerven,deLange,andHeskes(2010).Here,adeepbeliefnet-
Särkkäetal.,2012)anddata-drivenestimationofhemodynamic
work(DBN)ratherthanadeepfeed-forwardneuralnetworkwas
responsefunctionsthatmapneuralactivitytoobservedBOLDre-
|         |              |             |             |     |             |     |         | used to reconstruct |     | perceived | images |     | from fMRI | measurements. |     |
| ------- | ------------ | ----------- | ----------- | --- | ----------- | --- | ------- | ------------------- | --- | --------- | ------ | --- | --------- | ------------- | --- |
| sponses | will improve | sensitivity | (Pedregosa, |     | Eickenberg, |     | Ciuciu, |                     |     |           |        |     |           |               |     |
DBNsdifferfromtheirfeed-forwardcounterpartsbyhavingrecur-
| Thirion, | & Gramfort, | 2015). | Combining | data | across | subjects | via |     |     |     |     |     |     |     |     |
| -------- | ----------- | ------ | --------- | ---- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
rentconnectionsbetweennetworklayers.Moreover,theyaregen-
techniquessuchashyperalignment(Haxbyetal.,2011)mayalso
|     |     |     |     |     |     |     |     | erative models |     | in the sense | that | sensory | inputs | can be | sampled |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | ---- | ------- | ------ | ------ | ------- |
furtherimprovesensitivity,asdemonstratedinGüçlüandvanGer-
|         |             |        |         |              |     |         |        | from the | model. | This is | reminiscent | of  | predictive | processing, | in  |
| ------- | ----------- | ------ | ------- | ------------ | --- | ------- | ------ | -------- | ------ | ------- | ----------- | --- | ---------- | ----------- | --- |
| ven (in | press). The | use of | optimal | experimental |     | designs | (Lewi, |          |        |         |             |     |            |             |     |
whichtop-downpredictionsarethoughttoexplainawaybottom-
Butera,&Paninski,2009),wheretheoptimalstimulationsequence
upinputs(Clark,2016;Rao&Ballard,1999);aviewwhichwasal-
| is determined | on  | the fly such | as to | maximize | information |     | gain, |     |     |     |     |     |     |     |     |
| ------------- | --- | ------------ | ----- | -------- | ----------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
readypresentinthewritingsofThurstone(1923).Indeed,results
couldalsofurtherimproveresults.Todate,thisapproachremains
showthat,bysamplingfromtheDBNafterconditioningitsinternal
| underexplored | as  | it would | require | near real-time |     | estimation | of  |     |     |     |     |     |     |     |     |
| ------------- | --- | -------- | ------- | -------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
modelparametersandtheuseofclosed-loopexperimentalsetups. stateonobservedbraindata,accuratereconstructionsofperceived
handwrittendigitscouldbeobtained.
|     |     |     |     |     |     |     |     | Lateral | and feedback |     | connections |     | also support | the | inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ----------- | --- | ------------ | --- | --------- |
6.2. Improvingmodelrealism
|     |     |     |     |     |     |     |     | of long-term | temporal |     | dependencies. |     | These | dependencies | are |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --- | ------------- | --- | ----- | ------------ | --- |
essentialsincetheyaffordmodelingofcomplexspatio-temporal
Atpresent,encodingmodelsdonotyetprovideaparticularly
| veridical | mapping | from stimuli | to  | observed | brain | responses. | In  |     |     |     |     |     |     |     |     |
| --------- | ------- | ------------ | --- | -------- | ----- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
theircurrentformulation,varioussimplifyingassumptionsarein-
| troducedeitherforreasonsofcomputationalefficiencyorbecause |     |     |     |     |     |     |     | 10  |     |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ButseeBengio,Lee,Bornschein,andLin(2015)forsomerecentdevelopments
thedetailsunderlyingparticulartransformationsremainelusive.
whichmakesdeeplearningbasedonbackpropagationlessimplausible.

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 181
patternsinthesensoryinputandcanaccountforadaptationeffects Bialek,W.,Rieke,F.,deRuytervanSteveninck,R.,&Warland,D.(1991).Readinga
attheneuronallevel.Onewaytoincorporatepopulationdynamics neuralcode.Science,252,1854–1857.
Blakemore,C.,&Cooper,G.F.(1970).Developmentofthebraindependsonthe
inanencodingsettingisthroughtheuserecurrentneuralnetworks
visualenvironment.Nature,228(5270),477–478.
thathavepersistentstates(Hochreiter&Schmidhuber,1997).This Bliss,T.V.P.,&Lømo,T.(1973).Long-lastingpotentiationofsynaptictransmission
approachwasfollowedinGüçlüandvanGerven(2016)tomodel inthedentateareaoftheanaesthetizedrabbitfollowingstimulationofthe
perforantpath.JournalofPhysiology,232(2),357–374.
how low-level and high-level representations of dynamically
Brewer,A.A.,&Barton,B.(2014).VisualcortexinageingandAlzheimer’sdisease:
changinginputsinfluencetheBOLDresponse. changes in visual field maps and population receptive fields. Frontiers in
Given the rapid development of neural network theory, both Psychology,5(74),1–16.
intheengineeringandcomputationalneurosciencecommunities Brouwer, G. J., & Heeger, D. J. (2013). Categorical clustering of the neural
representationofcolor.JournalofNeuroscience,33(39),15454–15465.
(Yuste,2015),itistobeexpectedthatincreasinglysophisticated
Carandini,M.,Demb,J.B.,Mante,V.,Tolhurst,D.J.,Dan,Y.,Olshausen,B.A.,etal.
neural network architectures will become available. When one (2005).Doweknowwhattheearlyvisualsystemdoes?JournalofNeuroscience,
considerssuchanarchitectureasaputativeexplanationofneural 25(46),10577–10597.
Carlson,T.A.(2014).Orientationdecodinginhumanvisualcortex:newinsights
informationprocessingthentheencodingframeworkprovidesa
fromanunbiasedperspective.JournalofNeuroscience,34(24),8373–8383.
meanstotestthishypothesisagainstneuraldata. Chong,E.,Familiar,A.M.,&Shim,W.M.(2015).Reconstructingrepresentations
ofdynamicvisualobjectsinearlyvisualcortex.ProceedingsoftheNational
AcademyofSciences,113(5),1453–1458.
7. Conclusions Churchland, P. S., & Sejnowski, T. J. (1990). Neural representation and neural
computation.PhilosophicalPerspectives,4(1990),343–382.
Cichy,R.M.,Khosla,A.,Pantazis,D.,Torralba,A.,&Oliva,A.(2016).Deepneural
This review addressed the theoretical basis, applications and
networks predict hierarchical spatio-temporal cortical dynamics of human
future directions of encoding models in sensory neuroscience. visualobjectrecognition.arXiv:1601.02970[cs.CV].
These encoding models strive to provide new insights into the Cichy,R.,&Pantazis,D.(2015).Canvisualinformationencodedincorticalcolumns
functionalmechanismsthatmediatehowrepresentationsofour bedecodedfrommagnetoencephalographydatainhumans?Neuroimage,121,
193–204.
environmentareencodedinneuralresponsepatterns. Clark, A. (2016). Surfing uncertainty: prediction, action, and the embodied mind.
Encoding can be viewed as a model-based approach where OxfordUniversityPress.
Clarke,A.,Devereux,B.J.,Randall,B.,&Tyler,L.K.(2014).Predictingthetimecourse
internal latent states of computational models are fitted to
ofindividualobjectswithMEG.CerebralCortex,25,3602–3612.
neural and behavioral responses as subjects are engaged in that Corney,D.,&Lotto,R.B.(2007).Whatarelightnessillusionsandwhydowesee
particular task (Anderson et al., 2008; Ashby & Waldschmidt, them?PLoSComputationalBiology,3(9),e180.
2008;Forstmann&Wagenmakers,2015).Theadvantagesofthis Cox,D.D.,&Dean,T.(2014).Neuralnetworksandneuroscience-inspiredcomputer
vision.CurrentBiology,24(18),R921–R929.
approach are that (i) a good match between internal states and
Cox,D.D.,&Savoy,R.L.(2003).Functionalmagneticresonanceimaging(fMRI)
observed regional brain responses can be taken as evidence for ‘‘brainreading’’:detectingandclassifyingdistributedpatternsoffMRIactivity
a neural correlate of that state and (ii) different computational inhumanvisualcortex.Neuroimage,19,261–270.
Creutzfeldt, O. D., & Nothdurft, H. C. (1978). Representation of complex visual
modelscanbetestedagainsteachotherbycomparingtheirfitto
stimuliinthebrain.Naturwissenschaften,65(6),307–318.
observed data. In the context of sensory neuroscience, concrete Çukur,T.,Nishimoto,S.,Huth,A.G.,&Gallant,J.L.(2013).Attentionduringnatural
andabstractstimulusfeaturesplaytheroleoftheselatentstates. visionwarpssemanticrepresentationacrossthehumanbrain.Nature,16(6),
763–770.
Thecomparisonofencodingmodelsthatusedifferentunderlying
Daugman, J. G. (1985). Uncertainty relation for resolution in space, spatial
featurerepresentationsprovidethemeanstocomparealternative frequency,andorientationoptimizedbytwo-dimensionalvisualcorticalfilters.
theoriesaboutsensoryprocessinginthehumanbrain. JournaloftheOpticalSocietyofAmericaA,2(7),1160–1169.
David,S.V.,Hayden,B.Y.,Mazer,J.A.,&Gallant,J.L.(2008).Attentiontostimulus
In summary, encoding models that are grounded in theory
featuresshiftsspectraltuningofV4neuronsduringnaturalvision.Neuron,
andveridicallymodel thetransformationfromarbitrary sensory 59(3),509–521.
stimulitomeasuredneuralresponsepatternsprovidethemeans DeSimone,K.,Viviano,J.D.,&Schneider,K.A.(2015).Populationreceptivefield
estimation reveals new retinotopic maps in human subcortex. Journal of
tofurtherourunderstandingofneuralrepresentations.Ultimately,
Neuroscience,35(27),9836–9847.
thiscouldevenallowfaithfuldecodingofinternalrepresentations Edelman,S.(2015).Theminorityreport:somecommonassumptionstoreconsider
intheabsenceofsensorystimulation. inthemodellingofthebrainandbehaviour.JournalofExperimentalandTheoret-
icalArtificialIntelligence,http://dx.doi.org/10.1080/0952813X.2015.1042534.
Einhäuser,W.,&König,P.(2010).Gettingreal-sensoryprocessingofnatural
Acknowledgments stimuli.CurrentOpinioninNeurobiology,20(3),389–395.
Emadi, N., & Esteky, H. (2014). Behavioral demand modulates object category
representation in the inferior temporal cortex. Journal of Neurophysiology,
ThisresearchwassupportedbyVIDIgrantnumber639.072.513 112(10),2628–2637.
of The Netherlands Organization for Scientific Research (NWO). Felsen, G., & Dan, Y. (2005). A natural approach to studying vision. Nature
TheauthorexpresseshisgratitudetoSanderBosch,UmutGüçlü, Neuroscience,8(12),1643–1646.
Fitzpatrick,D.(2000).Seeingbeyondthereceptivefieldinprimaryvisualcortex.
MaxHinne,KatjaMüller,SilvanQuaxandthereviewersfortheir
CurrentOpinioninNeurobiology,10(4),438–443.
helpfulcomments. Forstmann,B.U.,&Wagenmakers,E.-J.(2015).Model-basedcognitiveneuroscience:
aconceptualintroduction.Springer.
Freeman, J., Brouwer, G. J., Heeger, D. J., & Merriam, E. P. (2011). Orientation
References decoding depends on maps, not columns. Journal of Neuroscience, 31,
4792–4804.
Frégnac,Y.,&Bathellier,B.(2015).Corticalcorrelatesoflow-levelperception:From
Ahmad,S.,&Yu,A.J.(2013).ActivesensingasBayes-optimalsequentialdecision
neuralcircuitstopercepts.Neuron,88(1),110–126.
making.InProc.twenty-ninthannu.conf.uncertain.artif.intell(pp.12–21).
Friston,K.J.,Holmes,A.P.,Worsley,K.J.,Poline,J.-B.,Frith,C.D.,&Frackowiak,R.
Anderson,J.,Carter,C.,Fincham,J.,Qin,Y.,Ravizza,S.,&Rosenberg-Lee,M.(2008).
S.J.(1995).Statisticalparametricmapsinfunctionalimaging:Agenerallinear
Using fMRI to test models of complex cognition. Cognitive Science, 32(8),
approach.HumanBrainMapping,2,189–210.
1323–1348.
Aquino,K.M.,Robinson,P.A.,&Drysdale,P.M.(2014).Spatiotemporalhemody- Fritz,J.,Shamma,S.,Elhilali,M.,&Klein,D.(2003).Rapidtask-relatedplasticity
namicresponsefunctionsderivedfromphysiology.JournalofTheoreticalBiol- of spectrotemporal receptive fields in primary auditory cortex. Nature
ogy,347,118–136. Neuroscience,6(11),1216–1223.
Ashby,F.,&Waldschmidt,J.(2008).FittingcomputationalmodelstofMRIdata. Fukushima,K.(1980).Neocognitron:Aself-organizingneuralnetworkmodelfor
BehaviorResearchMethods,40(3),713–721. amechanismofpatternrecognitionunaffectedbyshiftinposition.Biological
Barlow, H. B. (1953). Summation and inhibition in the frog’s retina. Journal of Cybernetics,36(4),193–202.
Physiology,119,69–88. Gallant,J.L.,Braun,J.,&VanEssen,D.C.(1993).Selectivityforpolar,hyperbolic,
Barth,M.,&Poser,B.A.(2011).Advancesinhigh-fieldBOLDfMRI.Materials,4(11), andCartesiangratingsinmacaquevisualcortex.Science,259(5091),100–103.
1941–1955. Gallant,J.L.,Connor,C.E.,&VanEssen,D.C.(1998).NeuralactivityinareasV1,V2
Bengio,Y.,Lee,D.-H.,Bornschein,J.,&Lin,Z.(2015).Towardsbiologicallyplausible andV4duringfreeviewingofnaturalscenescomparedtocontrolledviewing.
deeplearning.arXiv:1502.04156[cs.LG]. NeuroReport,9(9),2153–2158.

182 M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183
Gerstner,W.,Kistler,W.M.,Naud,R.,&Paninski,L.(2014).Neuronaldynamics: Khaligh-Razavi,S.-M.,&Kriegeskorte,N.(2014).Deepsupervised,butnotunsuper-
fromsingleneuronstonetworksandmodelsofcognition.Cambridge:Cambridge vised,modelsmayexplainITcorticalrepresentation.PLoSComputationalBiol-
UniversityPress. ogy,10(11),e1003915.
Gilbert,C.D.,&Li,W.(2013).Top-downinfluencesonvisualprocessing.Nature Klein,B.P.,Harvey,B.M.,&Dumoulin,S.O.(2014).Coordinatedattractionofspatial
ReviewsNeuroscience,14(5),350–363. response selectivity by spatial attention throughout visual cortex. Neuron,
Gollisch,T.,&Meister,M.(2010).Eyesmarterthanscientistsbelieved:neural 84(1),20.
computationsincircuitsoftheretina.Neuron,65(2),150–164. Knudsen,E.I.,&Konishi,M.(1978).Center–surroundorganizationofauditory
Güçlü, U., Thielen, J., Hanke, M., & van Gerven, M.A.J. (2016). Brains on beats. receptivefieldsintheowl.Science,202(4369),778–780.
arXiv:1606.02627[q-bio.NC]. Kobatake,E.,&Tanaka,K.(1994).Neuronalselectivitiestocomplexobjectfeatures
Güçlü,U.,&vanGerven,M.A.J.(2014).Unsupervisedfeaturelearningimproves in the ventral visual pathway of the macaque cerebral cortex. Journal of
prediction of human brain activity in response to natural images. PLoS Neurophysiology,7(3),856–867.
ComputationalBiology,10(8),e1003724. Kok,P.,Bains,L.J.,vanMourik,T.,Norris,D.G.,&deLange,F.P.(2016).Selective
Güçlü,U.,&vanGerven,M.A.J.(2015).Deepneuralnetworksrevealagradientin activationofthedeeplayersofthehumanprimaryvisualcortexbytop-down
thecomplexityofneuralrepresentationsacrosstheventralstream.Journalof feedback.CurrentBiology,26(3),371–376.
Neuroscience,35(27),10005–10014. Kok, P., & de Lange, F. P. (2014). Shape perception simultaneously up- and
Güçlü,U., &van Gerven, M.A. J.(2015).Increasinglycomplex representations downregulatesneuralactivityintheprimaryvisualcortex.CurrentBiology,
of natural movies across the dorsal stream are shared between subjects. 24(13),1531–1535.
Neuroimage,(inpress). Kriegeskorte,N.(2015).Deepneuralnetworks:Anewframeworkformodeling
Güçlü,U.,&vanGerven,M.A.J.(2016).Modelingthedynamicsofhumanbrain biologicalvisionandbraininformationprocessing.AnnualReviewofVision
activitywithrecurrentneuralnetworks.arXiv:1606.03071[q-bio.NC]. Science,1(1),417–446.
Hanke,M.,Baumgartner,F.J.,Ibe,P.,Kaule,F.R.,Pollmann,S.,Speck,O.,etal.(2014). Kriegeskorte,N.,Cusack,R.,&Bandettini,P.(2010).HowdoesanfMRIvoxelsample
Ahigh-resolution7-TeslafMRIdatasetfromcomplexnaturalstimulationwith the neuronal activity pattern: compact-kernel or complex spatio-temporal
anaudiomovie.ScientificData,1,1–18. filter?Neuroimage,49,1965–1976.
Harrison,S.A.,&Tong,F.(2009).Decodingrevealsthecontentsofvisualworking Kriegeskorte,N.,&Kreiman,G.(2011).Visualpopulationcodes.Cambridge,MA:The
memoryinearlyvisualareas.Nature,458,632–635. MITPress.
Hartline,H.K.(1940).Thereceptivefieldsofopticnervefibers.AmericanJournalof Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity
Physiology,130,690–699. analysis-connectingthebranchesofsystemsneuroscience.FrontiersinSystems
Hasson,U.,Malach,R.,&Heeger,D.J.(2010).Reliabilityofcorticalactivityduring Neuroscience,2(4).
naturalstimulation.TrendsinCognitiveSciences,14(1),40–48. Kuffler, S. W. S. W. (1953). Discharge patterns and functional organization of
Haxby,J.V.,Gobbini,M.I.,Furey,M.L.,Ishai,A.,Schouten,J.L.,&Pietrini,P.(2001). mammalianretina.JournalofNeurophysiology,16(1),37–68.
Distributedandoverlappingrepresentationsoffacesandobjectsinventral LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553),
temporalcortex.Science,293,2425–2430. 436–444.
Lee,S.,Papanikolaou,A.,Logothetis,N.K.,Smirnakis,S.M.,&Keliris,G.A.(2013).
Haxby,J.V.,Guntupalli,J.S.,Connolly,A.C.,Halchenko,Y.O.,Conroy,B.R.,Gobbini,
Anewmethodforestimatingpopulationreceptivefieldtopographyinvisual
M.I.,etal.(2011).Acommon,high-dimensionalmodeloftherepresentational
cortex.Neuroimage,81,144–157.
spaceinhumanventraltemporalcortex.Neuron,72(2),404–416.
Lewi,J.,Butera,R.,&Paninski,L.(2009).Sequentialoptimaldesignofneurophysi-
Haynes,J.-D.,&Rees,G.(2006).Decodingmentalstatesfrombrainactivityin
ologyexperiments.NeuralComputation,21(3),619–687.
humans.NatureReviewsNeuroscience,7,523–534.
Logothetis,N.K.(2008).WhatwecandoandwhatwecannotdowithfMRI.Nature,
Haynes,J.-D.,Sakai,K.,Rees,G.,Gilbert,S.,Frith,C.,&Passingham,R.E.(2007).
453(7197),869–878.
Readinghiddenintentionsinthehumanbrain.CurrentBiology,17(4),323–328.
Marčelja,S.(1980).Mathematicaldescriptionoftheresponsesofsimplecortical
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural
cells.JournaloftheOpticalSocietyofAmericaA,70(11),1297–1300.
Computation,9(8),1735–1780.
Marr,D.(1970).Atheoryforcerebralneocortex.ProceedingsoftheRoyalSocietyof
Horikawa,T.,&Kamitani,Y.(2016).Genericdecodingofseenandimaginedobjects
LondonBBiologicalSciences,176(1043),161–234.
usinghierarchicalvisualfeatures.arXiv:1510.06479[q-bio.NC].
Masland, R. H. (2012). The neuronal organization of the retina. Neuron, 76(2),
Horikawa,T.,Tamaki,M.,Miyawaki,Y.,&Kamitani,Y.(2013).Neuraldecodingof
266–280.
visualimageryduringsleep.Science,340(6132),639–642. McKee,J.L.,Riesenhuber,M.,Miller,E.K.,&Freedman,D.J.(2014).Taskdependence
Hsiao,S.(2008).Centralmechanismsoftactileshapeperception.CurrentOpinionin of visual and category representations in prefrontal and inferior temporal
Neurobiology,18(4),418–424. cortices.JournalofNeuroscience,34(48),16065–16075.
Hubel,D.H.,&Wiesel,T.N.(1959).Receptivefieldsofsingleneuronesinthecat’s Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Dis-
striatecortex.JournalofPhysiology,148,574–591. tributed representations of words and phrases and their compositionality.
Huth,A.G.,deHeer,W.A.,Griffiths,T.L.,&Gallant,J.L.(2016).Naturalspeech arXiv:1310.4546[cs.CL].
reveals the semantic maps that tile human cerebral cortex. Nature, 532, Mitchell,T.M.,Shinkareva,S.V.,Carlson,A.,Chang,K.-M.,Malave,V.L.,Mason,R.
453–458. A.,etal.(2008).Predictinghumanbrainactivityassociatedwiththemeanings
Huth,A.G.,Nishimoto,S.,Vu,A.T.,&Gallant,J.L.(2012).Acontinuoussemantic ofnouns.Science,320(5880),1191–1195.
spacedescribestherepresentationofthousandsofobjectandactioncategories Miyawaki, Y., Uchida, H., Yamashita, O., Sato, M., Morito, Y., Tanabe, H. C.,
acrossthehumanbrain.Neuron,76(6),1210–1224. etal.(2008).Visualimagereconstructionfromhumanbrainactivityusinga
Hyvärinen,A.,&Hoyer,P.O.(2001).Atwo-layersparsecodingmodellearnssimple combinationofmultiscalelocalimagedecoders.Neuron,60(5),915–929.
andcomplexcellreceptivefieldsandtopographyfromnaturalimages.Vision Mnih,V.,Kavukcuoglu,K.,Silver,D.,Rusu,A.a.,Veness,J.,Bellemare,M.G.,etal.
Research,41(18),2413–2423. (2015). Human-level control through deep reinforcement learning. Nature,
Johnson,M.H.,&Munakata,Y.(2005).Processesofchangeinbrainandcognitive 518(7540),529–533.
development.TrendsinCognitiveSciences,9(3),152–158. Monti,M.M.(2011).StatisticalanalysisoffMRItime-series:Acriticalreviewofthe
Jones,J.P.,&Palmer,L.A.(1987).Anevaluationofthetwo-dimensionalGaborfilter GLMapproach.FrontiersinHumanNeuroscience,5,28.
modelofsimplereceptivefieldsincatstriatecortex.JournalofNeurophysiology, Muckli,L.,DeMartino,F.,Vizioli,L.,Petro,L.S.,Smith,F.W.,Uğurbil,K.,etal.
58,1233–1258. (2015).ContextualfeedbacktosuperficiallayersofV1.CurrentBiology,25(20),
Joo,S.J.,Boynton,G.M.,&Murray,S.O.(2012).Long-range,pattern-dependent 2690–2695.
contextualeffectsinearlyhumanvisualcortex.CurrentBiology,22(9),781–786. Mumford,D.(1992).Onthecomputationalarchitectureoftheneocortex.II.The
Kamitani,Y.,&Tong,F.(2005).Decodingthevisualandsubjectivecontentsofthe roleofcortico-corticalloops.BiologicalCybernetics,66(3),241–251.
humanbrain.NatureNeuroscience,8(5),679–685. Naselaris,T.,Kay,K.N.,Nishimoto,S.,&Gallant,J.L.(2011).Encodinganddecoding
Kay,K.N.,Naselaris,T.,Prenger,R.J.,&Gallant,J.L.(2008).Identifyingnatural infMRI.Neuroimage,56(2),400–410.
imagesfromhumanbrainactivity.Nature,452,352–355. Naselaris,T.,Olman,C.A.,Stansbury,D.E.,Ugurbil,K.,&Gallant,J.L.(2015).A
Kay, K. N., Rokem, A., Winawer, J., Dougherty, R. F., & Wandell, B. A. (2013). voxel-wiseencodingmodelforearlyvisualareasdecodesmentalimagesof
GLMdenoise:Afast,automatedtechniquefordenoisingtask-basedfMRIdata. rememberedscenes.Neuroimage,105,215–228.
FrontiersinNeuroscience,7(247). Naselaris,T.,Prenger,R.J.,Kay,K.N.,Oliver,M.,&Gallant,J.L.(2009).Bayesian
Kay, K. N., Weiner, K. S., & Grill-Spector, K. (2015). Attention reduces spatial reconstructionofnaturalimagesfromhumanbrainactivity.Neuron,63(6),
ucertaintyinhumanventraltemporalcortex.CurrentBiology,25(5),595–600. 902–915.
Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive Ng,A.Y.,&Jordan,M.I.(2001).Ondiscriminativevs.generativeclassifiers:A
spatialsummationinhumanvisualcortex.JournalofNeurophysiology,110(2), comparisonoflogisticregressionandnaiveBayes.InAdv.neuralinf.process.
481–494. syst.,Vol.14(pp.1–8).
Kay,K.N.,Winawer,J.,Rokem,A.,Mezer,A.,&Wandell,B.A.(2013).Atwo-stage Niazi, A. M., van den Broek, P. L. C., Klanke, S., Barth, M., Poel, M., Desain, P.,
cascademodelofBOLDresponsesinhumanvisualcortex.PLoSComputational etal.(2014).Onlinedecodingofobject-basedattentionusingreal-timefMRI.
Biology,9(5),e1003079. EuropeanJournalofNeuroscience,30(2),319–329.
Kayser,C.,Körding,K.P.,&König,P.(2004).Processingofcomplexstimuliand Nishida, S., Huth, A.G., Gallant, J.L., & Nishimoto, S. (2015). Word statistics
natural scenes in the visual cortex. Current Opinion in Neurobiology, 14(4), in large-scale texts explain the human cortical semantic representation of
468–473. objects,actions,andimpressions.InThe45thannualmeetingofthesocietyfor
Kell,A.,Yamins,D.,Norman-Haignere,S.,Seibert,D.,Hong,H.,DiCarlo,J.,etal. neuroscience,Vol.333,no.13.
(2015).Computationalsimilaritiesbetweenvisualandauditorycortexstudied Nishimoto,S.,Vu,A.T.,Naselaris,T.,Benjamini,Y.,Yu,B.,&Gallant,J.L.(2011).
withconvolutionalneuralnetworks,fMRI,andelectrophysiology.Journalof Reconstructing visual experiences from brain activity evoked by natural
Visualization,15(12),1093. movies.CurrentBiology,21,1–6.

M.A.J.vanGerven/JournalofMathematicalPsychology76(2017)172–183 183
Norman,K.A.,Polyn,S.M.M.,Detra,G.J.,&Haxby,J.V.(2006).Beyondmind- Stanley,G.B.,Li,F.F.,&Dan,Y.(1999).Reconstructionofnaturalscenesfrom
reading:multi-voxelpatternanalysisoffMRIdata.TrendsinCognitiveSciences, ensembleresponsesinthelateralgeniculatenucleus.JournalofNeuroscience,
10(9),424–430. 19(18),8036–8042.
Okazawa,G.,Tajima,S.,&Komatsu,H.(2015).Imagestatisticsunderlyingnatural Stelzer,J.,Lohmann,G.,Mueller,K.,Buschmann,T.,&Turner,R.(2014).Deficient
texture selectivity of neurons in macaque V4. Proceedings of the National approachestohumanneuroimaging.FrontiersinHumanNeuroscience,8,462.
AcademyofSciences,112(4),E351–E360. Stokes,M.,Thompson,R.,Cusack,R.,&Duncan,J.(2010).Codesinvisualcortex
Olshausen,B.A.,&Field,D.J.(1996).Emergenceofsimple-cellreceptivefield duringmentalimagery.JournalofNeuroscience,29(2009),1565–1572.
propertiesbylearningasparsecodefornaturalimages.Nature,381,607–609. Summerfield, C., & de Lange, F. P. (2014). Expectation in perceptual decision
OpdeBeeck,H.P.(2010).Againsthyperacuityinbrainreading:spatialsmoothing making:neuralandcomputationalmechanisms.NatureReviewsNeuroscience,
doesnothurtmultivariatefMRIanalyses?Neuroimage,49,1943–1948. 15,745–756.
Orban,G.A.(2008).Higherordervisualprocessinginmacaqueextrastriatecortex. Swisher, J. D., Gatenby, J. C., Gore, J. C., Wolfe, B. A., Moon, C. H., Kim, S. G.,
PhysiologicalReviews,88(1),59–89. etal.(2010).Multiscalepatternanalysisoforientation-selectiveactivityinthe
Paninski, L. (2004). Maximum likelihood estimation of cascade point-process primaryvisualcortex.JournalofNeuroscience,30,325–330.
neuralencodingmodels.NetworkComputationinNeuralSystems,15,243–262. Thirion,B.,Duchesnay,E.,Hubbard,E.,Dubois,J.,Poline,J.-B.,Lebihan,D.,etal.
Pantazis,D.,Oliva,A.,&Cichy,R.M.(2014).Resolvinghumanobjectrecognitionin
(2006).Inverseretinotopy:inferringthevisualcontentofimagesfrombrain
spaceandtime.NatureNeuroscience,17,455–462.
activationpatterns.Neuroimage,33(4),1104–1116.
Parkkonen,L.,Andersson,J.,Hämäläinen,M.,&Hari,R.(2008).Earlyvisualbrain
Thurstone,L.L.(1923).Thestimulus–responsefallacyinpsychology.Psychological
areasreflecttheperceptofanambiguousscene.ProceedingsoftheNational
Review,3,354–369.
AcademyofSciences,105(51),20500–20504.
Tong,F.,&Pratte,M.M.S.(2012).Decodingpatternsofhumanbrainactivity.Annual
Pedregosa,F.,Eickenberg,M.,Ciuciu,P.,Thirion,B.,&Gramfort,A.(2015).Data-
ReviewofPsychology,63,483–509.
drivenHRFestimationforencodinganddecodingmodels.Neuroimage,104,
vanBergen,R.,Ma,W.J.,Pratte,M.S.,&Jehee,J.F.M.(2015).Sensoryuncertainty
209–220.
Peirce,J.W.(2015).Understandingmid-levelrepresentationsinvisualprocessing. decoded from visual cortex predicts behavior. Nature Neuroscience, 18(12),
JournalofVisualization,15(7),5. 1728–1730.
vandeNieuwenhuijzen,M.E.,Backus,A.R.,Bahramisharif,A.,Doeller,C.F.,Jensen,
Polanía,R.,Paulus,W.,&Nitsche,M.A.(2012).Noninvasivelydecodingthecontents
ofvisualworkingmemoryinthehumanprefrontalcortexwithinhigh-gamma O.,&vanGerven,M.A.J.(2013).MEG-baseddecodingofthespatiotemporal
oscillatorypatterns.JournalofCognitiveNeuroscience,24(2),304–314. dynamicsofvisualcategoryperception.Neuroimage,83,1063–1073.
Purves, D., Wojtach, W. T., & Lotto, R. B. (2011). Understanding vision in vanGerven,M.A.J.,deLange,F.P.,&Heskes,T.(2010).Neuraldecodingwith
whollyempiricalterms.ProceedingsoftheNationalAcademyofSciences,108, hierarchicalgenerativemodels.NeuralComputation,22(12),3127–3142.
15588–15595. vanGerven,M.A.J.,Farquhar,J.,Schaefer,R.,Vlek,R.,Geuze,J.,Nijholt,A.,etal.
QuianQuiroga,R.,Reddy,L.,Kreiman,G.,Koch,C.,&Fried,I.(2005).Invariant (2009).Thebrain-computerinterfacecycle.JournalofNeuralEngineering,6(4),
visualrepresentationbysingleneuronsinthehumanbrain.Nature,435(7045), 41001.
1102–1107. vanGerven,M.A.J.,Kok,P.,deLange,F.P.,&Heskes,T.(2011).Dynamicdecoding
Rainer,G.,Lee,H.,&Logothetis,N.K.(2004).Theeffectoflearningonthefunction ofongoingperception.Neuroimage,57,950–957.
ofmonkeyextrastriatevisualcortex.PLoSBiology,2,e44. Vetter,P.,Smith,F.W.,&Muckli,L.(2014).Decodingsoundandimagerycontentin
Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a earlyvisualcortex.CurrentBiology,24,1–7.
functionalinterpretationofsomeextra-classicalreceptive-fieldeffects.Nature Victor,J.D.,Purpura,K.,Katz,E.,&Mao,B.(1994).Populationencodingofspatial
Neuroscience,2(1),79–87. frequency,orientation,andcolorinmacaqueV1.JournalofNeurophysiology,
Riesenhuber,M.,&Poggio,T.(1999).Hierarchicalmodelsofobjectrecognitionin 72(5),2151–2166.
cortex.NatureNeuroscience,2(11),1019–1025. Vu, V., Ravikumar, P., & Naselaris, T. (2011). Encoding and decoding V1 fMRI
Ringach,D.(2004).Mappingreceptivefieldsinprimaryvisualcortex.Journalof responses to natural images with sparse nonparametric models. Annals of
Physiology,558(3),717–728. AppliedStatistics,5(2B),1159–1182.
Robinson,L.,&Rolls,E.T.(2015).Invariantvisualobjectrecognition:biologically Wandell,B.A.,&Winawer,J.(2011).Imagingretinotopicmapsinthehumanbrain.
plausibleapproaches.BiologicalCybernetics,209(4–5),505–535. VisionResearch,51(7),718–737.
Rust,N.C.,&Movshon,J.A.(2005).Inpraiseofartifice.NatureNeuroscience,8(12), Wandell, B. A., Winawer, J., & Kay, K. N. (2015). Computational modeling
1647–1650.
of responses in human visual cortex. In A. W. Toga (Ed.), Brain mapp.
Sanger,T.D.(2003).Neuralpopulationcodes.CurrentOpinioninNeurobiology,
(pp.651–659).AcademicPress.
13(2),238–249.
Williams,M.A.,Dang,S.,&Kanwisher,N.G.(2007).Onlysomespatialpatterns
Särkkä,S.,Solin,A.,Nummenmaa,A.,Vehtari,A.,Auranen,T.,Vanni,S.,etal.(2012).
DynamicretrospectivefilteringofphysiologicalnoiseinBOLDfMRI:DRIFTER. offMRIresponsearereadoutintaskperformance.NatureNeuroscience,10(6),
Neuroimage,60(2),1517–1527. 685–686.
Wilson,D.A.(2001).Receptivefieldsintheratpiriformcortex.ChemicalSenses,
Schoenmakers, S., Barth, M., Heskes, T., & van Gerven, M. A. J. (2013). Linear
26(5),577–584.
reconstructionofperceivedimagesfromhumanbrainactivity.Neuroimage,83,
Wu, M. M. C.-K., David, S. V. S., & Gallant, J. L. J. (2006). Complete functional
951–961.
Schoenmakers,S.,Güçlü,U.,vanGerven,M.A.J.,&Heskes,T.(2015).Gaussian characterizationofsensoryneuronsbysystemidentification.AnnualReviewof
mixturemodelsandsemanticgatingimprovereconstructionsfromhuman Neuroscience,29,477–505.
brainactivity.FrontiersinComputationalNeuroscience,8,1–10. Wundt,W.(1862).BeiträgezurTheoriederSinneswahrnehmung(Contributionson
Scholte,H.S.,Ghebreab,S.,Waldorp,L.,Smeulders,A.W.M.,&Lamme,V.A.F. theTheoryofSensoryPerception).Winter,Leipzig.
(2009).BrainresponsesstronglycorrelatewithWeibullimagestatisticswhen Yacoub, E., Harel, N., & Ugurbil, K. (2008). High-field fMRI unveils orientation
processingnaturalimages.JournalofVision,9(4),29.1–15. columnsinhumans.ProceedingsoftheNationalAcademyofSciences,105(30),
Serences,J.T.,Ester,E.F.,Vogel,E.K.,&Awh,E.(2009).Stimulus-specificdelay 10607–10612.
activityinhumanprimaryvisualcortex.PsychologicalScience,20(2),207–214. Yamins,D.L.K.,Hong,H.,Cadieu,C.F.,Solomon,E.A.,Seibert,D.,&DiCarlo,J.J.
Serre,T.,Wolf,L.,Bileschi,S.,Riesenhuber,M.,&Poggio,T.(2007).Robustobject (2014).Performance-optimizedhierarchicalmodelspredictneuralresponses
recognitionwithcortex-likemechanisms.IEEETransactionsonPatternAnalysis inhighervisualcortex.ProceedingsoftheNationalAcademyofSciences,111(23),
andMachineIntelligence,29(3),411–426. 8619–8624.
Sherrington,C.S.(1906).Theintegrativeactionofthenervoussystem.NewHaven, Yuste,R.(2015).Fromtheneurondoctrinetoneuralnetworks.NatureReviews
CT:YaleUniversityPress. Neuroscience,16(8),487–497.
Simoncelli,E.P.,&Olshausen,B.A.(2001).Naturalimagestatisticsandneural Zipser,K.,Lamme,V.A.F.,&Schiller,P.H.(1996).Contextualmodulationinprimary
representation.AnnualReviewofNeuroscience,24,1193–1216. visualcortex.JournalofNeuroscience,16(22),7376–7389.
Smith,F.W.,&Muckli,L.(2010).Nonstimulatedearlyvisualareascarryinformation Zuiderbaan,W.,Harvey,B.M.,&Dumoulin,S.O.(2012).Modelingcenter–surround
aboutsurroundingcontext.ProceedingsoftheNationalAcademyofSciences, configurationsinpopulationreceptivefieldsusingfMRI.JournalofVisualization,
107(46),20099–20103. 12(3),1–15.