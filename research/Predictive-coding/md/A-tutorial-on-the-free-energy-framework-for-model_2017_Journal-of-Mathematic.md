JournalofMathematicalPsychology76(2017)198–211
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
A tutorial on the free-energy framework for modelling perception
and learning
∗
RafalBogacz
MRCUnitforBrainNetworkDynamics,UniversityofOxford,MansfieldRoad,Oxford,OX13TH,UK
NuffieldDepartmentofClinicalNeurosciences,UniversityofOxford,JohnRadcliffeHospital,Oxford,OX39DU,UK
h i g h l i g h t s
• Bayesianinferenceaboutstimuluspropertiescanbeperformedbynetworksofneurons.
• LearningaboutstatisticsofstimulicanbeachievedbyHebbiansynapticplasticity.
• Structureofthemodelresemblesthehierarchicalorganizationoftheneocortex.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Thispaperprovidesaneasytofollowtutorialonthefree-energyframeworkformodellingperception
Availableonline14December2015 developedbyFriston,whichextendsthepredictivecodingmodelofRaoandBallard.Thesemodelsassume
thatthesensorycortexinfersthemostlikelyvaluesofattributesorfeaturesofsensorystimulifrom
thenoisyinputsencodingthestimuli.Remarkably,thesemodelsdescribehowthisinferencecouldbe
implementedinanetworkofverysimplecomputationalelements,suggestingthatthisinferencecouldbe
performedbybiologicalnetworksofneurons.Furthermore,learningabouttheparametersdescribingthe
featuresandtheiruncertaintyisimplementedinthesemodelsbysimplerulesofsynapticplasticitybased
onHebbianlearning.Thistutorialintroducesthefree-energyframeworkusingverysimpleexamples,and
providesstep-by-stepderivationsofthemodel.Italsodiscussesinmoredetailhowthemodelcouldbe
implementedinbiologicalneuralcircuits.Inparticular,itpresentsanextendedversionofthemodelin
whichtheneuronsonlysumtheirinputs,andsynapticplasticityonlydependsonactivityofpre-synaptic
andpost-synapticneurons.
©2015TheAuthor.PublishedbyElsevierInc.
ThisisanopenaccessarticleundertheCCBYlicense
(http://creativecommons.org/licenses/by/4.0/).
1. Introduction plasticity,andRaoandBallard(1999)demonstratedthatthemodel
presentedwithnaturalimageslearnsfeaturesresemblingrecep-
ThemodelofFriston(2005)andthepredictivecodingmodel tivefieldsofneuronsintheprimaryvisualcortex.
ofRaoandBallard(1999)provideapowerfulmathematicalframe- Friston(2005)hasextendedthemodeltoalsorepresentuncer-
worktodescribehowthesensorycortexextractsinformationfrom tainty associated with different features. He showed that learn-
noisystimuli.Thepredictivecodingmodel(Rao&Ballard,1999) ing about the variance and co-variance of features can also be
implementedbysimplesynapticplasticityrulesbasedonHebbian
suggeststhatvisualcortexinfersthemostlikelypropertiesofstim-
learning.Astheextendedmodel(Friston,2005)learnsthevariance
uli from noisy sensory input. The inference in this model is im-
andco-varianceoffeatures,itoffersseveralnewinsights.First,it
plementedbyasurprisinglysimplenetworkofneuron-likenodes.
describes how the perceptual systems may differentially weight
Themodeliscalled‘‘predictivecoding’’,becausesomeofthenodes
sourcesofsensoryinformationdependingontheirlevelofnoise.
inthenetworkencodethedifferencesbetweeninputsandpredic-
Second, it shows how the sensory networks can learn to recog-
tionsofthenetwork.Remarkably,learningaboutfeaturespresent
nize features that are encoded in the patterns of covariance be-
in sensory stimuli is implemented by simple Hebbian synaptic
tweeninputs,suchastextures.Third,itprovidesanaturalwayto
implementattentionalmodulationasthereductioninvarianceof
theattendedfeatures(wecomebacktotheseinsightsinDiscus-
∗ sion).Furthermore,Friston(2005)pointedoutthatthismodelcan
Correspondenceto:NuffieldDepartmentofClinicalNeurosciences,University
ofOxford,JohnRadcliffeHospital,Oxford,OX39DU,UK. beviewedasanapproximateBayesianinferencebasedonmini-
E-mailaddress:rafal.bogacz@ndcn.ox.ac.uk. mizationofafunctionreferredtoinstatisticsasfree-energy.The
http://dx.doi.org/10.1016/j.jmp.2015.11.003
0022-2496/©2015TheAuthor.PublishedbyElsevierInc.ThisisanopenaccessarticleundertheCCBYlicense(http://creativecommons.org/licenses/by/4.0/).

|     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     | 199 |
| --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
free-energyframework(Friston,2003,2005)hasbeenrecentlyex- architectures.Thetutorialprovidesstep-by-stepderivationofthe
tendedbyKarlFristonandhiscolleaguestodescribehowthebrain model.Someofthesederivationsarestraightforward,andwefeel
performs different cognitive functions including action selection thatitwouldbehelpfulforthereadertodothemontheirownto
(FitzGerald, Schwartenbeck, Moutoussis, Dolan, & Friston, 2015; gainabetterunderstandingofthemodelandto‘‘keepinmind’’
Fristonetal.,2013).Furthermore,Friston(2010)proposedthatthe thenotationusedinthepaper.Suchstraightforwardderivations
free-energytheoryunifiesseveraltheoriesofperceptionandac- areindicatedby‘‘(TRYITYOURSELF)’’,soafterencounteringsuch
tionwhicharecloselyrelatedtothefree-energyframework. labelwerecommendtryingtodothecalculationdescribedinthe
There are many articles which provide an intuition for the sentence with this label and then compare the obtained results
free-energy framework and discuss how it relates with other withthoseinthepaper.Toillustratethemodelweincludesimple
theoriesandexperimentaldata(Friston,2003,2005,2010;Friston simulations, but again we feel it would be helpful for a reader
et al., 2013). However, the description of mathematical details toperformthemontheirown,togetanintuitionforthemodel.
ofthetheoryinthesepapersrequiresaverydeepmathematical Thereforewedescribethesesimulationsasexercises.
background.Themaingoalofthispaperistoprovideaneasyto The paper is organized as follows. Section 2 introduces the
followtutorialonthefree-energyframework.Tomakethetutorial modelusingaverysimpleexampleusingasbasicmathematical
accessible to a wide audience, it only assumes basic knowledge conceptsaspossible,soitisaccessibletoaparticularlywideaudi-
ofprobabilitytheory,calculusandlinearalgebra.Thistutorialis ence.Section3providesmathematicalfoundationsforthemodel,
plannedtobecomplementarytoexistingliteraturesoitdoesnot andshowshowtheinferenceinthemodelisrelatedtominimiza-
focusontherelationshiptoothertheoriesandexperimentaldata, tionoffree-energy.Section4thenshowshowthemodelscalesup
and on applications to more complex tasks which are described todescribetheneuralcircuitsinsensorycortex.Inthesethreesec-
elsewhere(Friston,2010;Fristonetal.,2013). tionsweusenotationsimilartothatusedbyFriston(2005).Sec-
In this tutorial we also consider in more detail the neural tion5describesanextendedversionofthemodelwhichsatisfies
implementationofthefree-energyframework.Anycomputational theconstraintoflocalplasticitydescribedabove.Finally,Section6
model would need to satisfy the following constraints to be discussesinsightsprovidedbythemodel.
consideredbiologicallyplausible:
2. Simplestexampleofperception
1. Localcomputation:Aneuronperformscomputationsonlyon
| the basis | of the activity | of its | input neurons |     | and synaptic |          |                |         |                             |     |
| --------- | --------------- | ------ | ------------- | --- | ------------ | -------- | -------------- | ------- | --------------------------- | --- |
|           |                 |        |               |     |              | We start | by considering | in this | section a simple perceptual |     |
weightsassociatedwiththeseinputs(ratherthaninformation
probleminwhichavalueofasinglevariablehastobeinferredfrom
encodedinotherpartsofthecircuit).
asingleobservation.Tomakeitmoreconcrete,considerasimple
2. Localplasticity:Synapticplasticityisonlybasedontheactivity
ofpre-synapticandpost-synapticneurons. organism that tries to infer the size or diameter of a food item,
v
|           |            |         |        |                 |       | whichwedenoteby |     | ,onthebasisoflightintensityitobserves. |     |     |
| --------- | ---------- | ------- | ------ | --------------- | ----- | --------------- | --- | -------------------------------------- | --- | --- |
| The model | of Rao and | Ballard | (1999) | fully satisfied | these |                 |     |                                        |     |     |
Letusassumethatoursimpleanimalhasonlyonelightsensitive
constraints.ThemodelofFriston(2005)didnotsatisfythemfully,
receptorwhichprovidesitwithanoisyestimateoflightintensity,
butweshowthataftersmallmodificationsandextensionsitcan
whichwedenotebyu.Letgdenoteanon-linearfunctionrelating
| satisfy them. | So the descriptions |     | of the model | in  | this tutorial |     |     |     |     |     |
| ------------- | ------------------- | --- | ------------ | --- | ------------- | --- | --- | --- | --- | --- |
theaveragelightintensitywiththesize.Sincetheamountoflight
slightlydifferinafewplacesorextendtheoriginalmodeltobetter reflectedisrelatedtotheareaofanobject,inthisexamplewewill
explainhowtheproposedcomputationcouldbeimplementedin consider a simple function of g (v) = v2. Let us further assume
theneuralcircuits.Allsuchdifferencesorextensionsareindicated thatthesensoryinputisnoisy—inparticular,whenthesizeoffood
by footnotes or in text, and the original model is presented in itemis v ,theperceivedlightintensityisnormallydistributedwith
AppendixA. meang (v) ,andvarianceΣ (althoughanormaldistributionisnot
u
Itiscommonlyassumedintheoreticalneuroscience,(O’Reilly&
thebestchoiceforadistributionoflightintensity,asitincludes
Munakata,2000)thatthebasiccomputationsaneuronperforms
negativenumbers,wewillstilluseitforasimplicity):
| are the summation | of its | input weighted |     | by the | strengths of |          |             |     |     |     |
| ----------------- | ------ | -------------- | --- | ------ | ------------ | -------- | ----------- | --- | --- | --- |
|                   |        |                |     |        |              | ( u|v)=f | ( u;g (v),Σ | ).  |     |     |
synapticconnections,andthetransformationofthissumthrough p (1)
u
| a (monotonic) | function describing    |       | the relationship |              | between |           |                 |                                        |     |     |
| ------------- | ---------------------- | ----- | ---------------- | ------------ | ------- | --------- | --------------- | -------------------------------------- | --- | --- |
|               |                        |       |                  |              |         | InEq.(1)f | ( x;µ,Σ)        | denotesthedensityofanormaldistribution |     |     |
| neurons’      | total input and output | (also | termed           | firing-Input | or f-I  |           |                 |                                        |     |     |
|               |                        |       |                  |              |         | withmean  | µ andvarianceΣ: |                                        |     |     |
curve).Wheneverpossible,wewillassumethatthecomputation
 ( x−µ)2
| of the neurons                                     | in the described |     | model | is limited | to these |             | 1     |     |     |     |
| -------------------------------------------------- | ---------------- | --- | ----- | ---------- | -------- | ----------- | ----- | --- | --- | --- |
|                                                    |                  |     |       |            |          | f ( x;µ,Σ)= | √ exp | −   | .   | (2) |
| computations(orevenjusttolinearsummationofinputs). |                  |     |       |            |          |             | πΣ    | 2Σ  |     |     |
2
Wefeelthattheneuralimplementationofthemodelisworth
considering,becauseifthefree-energyprincipleindeeddescribes Due to the noise present in the observed light intensity, the
v
the computations in the brain, it can provide an explanation for animalcanrefineitsguessforthesize bycombiningthesensory
why the cortex is organized in a particular way. However to stimulus with the prior knowledge on how large the food items
gain such insight it is necessary to start comparing the neural usuallyare,thatithadlearntfromexperience.Forsimplicity,letus
networks implementing the model with those in the real brain. assumethatouranimalexpectsthissizetobenormallydistributed
Consequently, we consider in this paper possible neural circuits with mean v and variance Σ (subscript p stands for ‘‘prior’’),
|     |     |     |     |     |     |     | p   | p   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
whichwecanwriteas:
| that could | perform the computations |     | required | by  | the theory. |     |     |     |     |     |
| ---------- | ------------------------ | --- | -------- | --- | ----------- | --- | --- | --- | --- | --- |
Althoughtheneuralimplementationsproposedherearenotthe (v)=f (v;v ,Σ ).
|     |     |     |     |     |     | p   | p p |     |     | (3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onlypossibleones,itisworthconsideringthemasastartingpoint
forcomparisonofthemodelwithdetailsofneuralarchitecturesin Letusnowassumethatouranimalobservedaparticularvalue
thebrain.Wehopethatsuchcomparisoncoulditerativelyleadto oflightintensity,andattemptstoestimatethesizeofthefooditem
refinedneuralimplementationsthataremoreandmoresimilarto on the basis of this observation. We will first consider an exact
realneuralcircuits. solutiontothisproblem,andillustratewhyitwouldbedifficult
Tomakethistutorialaseasytofollowaspossibleweintroduce tocomputeitinasimpleneuralcircuit.Thenwewillpresentan
the free-energy framework using a simple example, and then approximatesolutionthatcanbeeasilyimplementedinasimple
illustrate how the model can scale up to more complex neural networkofneurons.

| 200 |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |             |                   |     |               |             |              |               |                   |           |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | ----------- | ----------------- | --- | ------------- | ----------- | ------------ | ------------- | ----------------- | --------- |
|     |     |     |     |                                                         |     |     | a           | few summary       |     | statistics    | like        | mean         | and variance. | Second,           | the       |
|     |     |     |     |                                                         |     |     | computation |                   | of  | the posterior |             | distribution | involves      | computation       |           |
|     |     |     |     |                                                         |     |     | of          | the normalization |     | term.         | Although    |              | it has        | been proposed     | that      |
|     |     |     |     |                                                         |     |     | circuits    | within            | the | basal         | ganglia     | can compute  |               | the normalization |           |
|     |     |     |     |                                                         |     |     | term        | in case           | of  | the discrete  | probability |              | distributions |                   | (Bogacz & |
Gurney,2007),computationofthenormalizationforcontinuous
distributionsinvolvesevaluatingtheintegralofEq.(5).Calculating
suchintegralwouldbechallengingforasimplebiologicalsystem.
|     |     |     |     |     |     |     | This | is especially |     | true | when the | dimensionality |     | of the | integrals |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------------- | --- | ---- | -------- | -------------- | --- | ------ | --------- |
(i.e.,thenumberofunknownvariables)increasesbeyondatrivial
|     |     |     |     |     |     |     | number. |     | Even mathematicians |     |     | resort | to (computationally |     | very |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------------------- | --- | --- | ------ | ------------------- | --- | ---- |
expensive)numericalorsamplingtechniquesinthiscase.
|     |     |     |     |     |     |     |     | We will | now | present | an approximate |     | solution | to  | the above |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------- | -------------- | --- | -------- | --- | --------- |
inferenceproblem,thatcouldbeeasilyimplementedinasimple
biologicalsystem.
|     |     |     |     |     |     |     | 2.2. | Findingthemostlikelyfeaturevalue |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Fig.1. Theposteriorprobabilityofthesizeofthefooditemintheproblemgiven
| inExercise1. |     |     |     |     |     |     |     |                                                |     |     |     |     |     | (v|u | )      |
| ------------ | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | ---- | ------ |
|              |     |     |     |     |     |     |     | Insteadoffindingthewholeposteriordistributionp |     |     |     |     |     |      | ,letus |
v
2.1. Exactsolution trytofindthemostlikelysizeofthefooditem whichmaximizes
|     |     |     |     |     |     |     | (v|u | )                                 |     |     |        |     |     | φ                |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --------------------------------- | --- | --- | ------ | --- | --- | ---------------- | --- |
|     |     |     |     |     |     |     | p    | .Wewilldenotethismostlikelysizeby |     |     |        |     |     | ,anditsposterior |     |
|     |     |     |     |     |     |     |      |                                   |     |     | (φ|u ) |     |     |                  |     |
Tocomputehowlikelydifferentsizes v aregiventheobserved probability density by p . It is reasonable to assume that in
manycasesthebrainrepresentsatagivenmomentoftimeonly
sensoryinputu,wecoulduseBayes’theorem:
|     |     |     |     |     |     |     | most | likely | values | of features. |     | For example |     | in case of | binocular |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------ | ------ | ------------ | --- | ----------- | --- | ---------- | --------- |
(v) ( u|v)
p p rivalry, only one of the two possible interpretations of sensory
| p (v|u )= |     | .   |     |     |     | (4) |                      |     |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|           | (   | )   |     |     |     |     | inputsisrepresented. |     |     |     |     |     |     |     |     |
p u
|     |     |     |     |     |     |     |     | Wewilllookforthevalue |     |     | φ whichmaximizesp |     |     | (φ|u ) | .According |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | ----------------- | --- | --- | ------ | ---------- |
|     | ( ) |     |     |     |     |     |     |                       |     |     |                   |     |     |        |            |
Termp u inthedenominatorofEq.(4)isanormalizationterm, toEq.(4),theposteriorprobabilityp (φ|u ) dependsonaratioof
|     |     |     |     |     |     | (v|u ) |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
which ensures that the posterior probabilities of all sizes p twoquantities,butthedenominatorp ( u ) doesnotdependon φ .
| integrateto1: |     |     |     |     |     |     |                |     |     | φ               |     |     | (φ|u )            |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --------------- | --- | --- | ----------------- | --- | --- |
|               |     |     |     |     |     |     | Thusthevalueof |     |     | whichmaximizesp |     |     | isthesameonewhich |     |     |

maximizesthenumeratorofEq.(4).Wewilldenotethelogarithm
| p ( u )= | p (v) | p ( u|v) d | v.  |     |     | (5) |     |     |     |     |     |     |     |     |     |
| -------- | ----- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ofthenumeratorbyF,asitisrelatedtothenegativeoffreeenergy
(aswewilldescribeinSection3):
Theintegralintheaboveequationsumsoverthewholerange
|                    |     |     |                                        |     |     |     | =lnp | (φ)+lnp |     | ( u|φ). |     |     |     |     |     |
| ------------------ | --- | --- | -------------------------------------- | --- | --- | --- | ---- | ------- | --- | ------- | --- | --- | --- | --- | --- |
|                    |     | v   |                                        |     |     |     | F    |         |     |         |     |     |     |     | (6) |
| ofpossiblevaluesof |     |     | ,soitisadefiniteintegral,butforbrevity |     |     |     |      |         |     |         |     |     |     |     |     |
ofnotationwedonotstatethelimitsofintegrationinthisandall In the above equation we used the property of logarithm
|                           |     |     |     |     |     |     | (   | )=lna+lnb.Wewillmaximizethelogarithmofthenumer- |     |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| otherintegralsinthepaper. |     |     |     |     |     |     | ln  | ab                                              |     |     |     |     |     |     |     |
NowcombiningEqs.(1)–(5)wecancomputenumericallyhow atorofEq.(4),becauseithasthesamemaximumasthenumerator
likelydifferentsizesaregiventhesensoryobservation.Forreaders itselfaslnisamonotonicfunction,andiseasiertocomputeasthe
|     |     |     |     |     |     |     |     |     |     | ( u|φ) | (φ) |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
whoarenotfamiliarwithsuchBayesianinferencewerecommend expressionsforp andp involveexponentiation.
doingthefollowingexercisenow. Tofindtheparameter φ thatdescribesthemostlikelysizeofthe
fooditem,wewilluseasimplegradientascent:i.e.wewillmodify
Exercise1. Assumethatouranimalobservedthelightintensityu= φ proportionallytothegradientofF,whichwillturnouttobea
2, the level of noise in its receptor is Σ = 1, and the mean and verysimpleoperation.Itisrelativelystraightforwardtocompute
u
varianceofitspriorexpectationofsizeare v =3andΣ =1.Write F bysubstitutingEqs.(1)–(3)intoEq.(6)andthentocomputethe
|     |     |     |     | p   | p   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
acomputerprogramthatcomputestheposteriorprobabilitiesofsizes
|     |     |     |     |     |     |     | derivativeofF |     | (TRYITYOURSELF). |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------------- | --- | --- | --- | --- | --- | --- |
from001to5,andplotsthem. .
|     |     |     |     |     |     |     |     | = (φ;v | ,Σ  | )+lnf | ( u;g | (φ),Σ | )   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | ----- | ----- | ----- | --- | --- | --- |
|     |     |     |     |     |     |     | F   | lnf    | p   | p     |       | u     |     |     |     |
TheMatlabcodeperformingthiscalculationisgivenattheend   (φ−v )2
1
o f th e p a pe r, a n d t he r e s u l t in g p l o t i s s h o w n i n Fi g . 1 . I t i s w o r th = ln exp − p
|     |     |     |     |     |     |     |     |     |  πΣ |     | 2Σ  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
o bs er v in g th a t su c h Ba y e s i a n a p p r o a c h i n t eg ra t es t h e i n f o r m a ti o n 2 p p
broughtbythestimu luswithpriorknowledge:pleasenotethatthe   ( (φ))2
|     |     | v   |     |     |     |     |     |     | 1   |     | u−g |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
most√likelyvalueof liesbetweenthatsuggestedbythestimulus + ln √ exp −
|                                                         |     |     |     |     |     |     |     |     | πΣ  |     |     | 2Σ  |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (i.e. 2)andthemostlikelyvaluebasedonpriorknowledge(i.e. |     |     |     |     |     |     |     |     | 2   |     |     | u   |     |     |     |
u
3).Itmayseemsurprisingwhytheposteriorprobabilityissolow (φ−v )2
|     |     |     |     |     |     |     |     |     | 1   | 1 lnΣ |     | p   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
f o r v = 3, i.e . th e m ea n p r io r e x p e c ta tio n . I t co m e s f r om t he f a c t = ln√ − −
|            |     |                                            |                     |                      |             |         |     |       | π   | 2 p   | 2Σ  |           |     |     |     |
| ---------- | --- | ------------------------------------------ | ------------------- | -------------------- | ----------- | ------- | --- | ----- | --- | ----- | --- | --------- | --- | --- | --- |
| (          | ) = |                                            |                     |                      |             | =       |     |       | 2   |       |     | p         |     |     |     |
| t h at g 3 | 9 , | w h ich                                    | i sr e a ll y f a r | fr o m o b s e rv ed | v a l u e u | 2 , s o |     |       |     |       |     |           |     |     |     |
| ( = 2|v    | =   | )                                          |                     |                      |             |         |     |       | 1   | 1     | (   | u−g (φ))2 |     |     |     |
| p u        |     | 3 isveryclosetozero.Thisillustrateshownon- |                     |                      |             |         |     | + ln√ |     | − lnΣ | −   |           |     |     |     |
|            |     |                                            |                     |                      |             |         |     |       |     |       | u   | 2Σ        |     |     |     |
intuitiveBayesianinferencecanbeoncetherelationshipbetween 2 π 2
u
| variablesisnon-linear. |     |     |     |     |     |     |     |    |     |      |     |     |       | (φ))2 |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | ----- | ------ | --- |
|                        |     |     |     |     |     |     |     | 1   |     | (φ−v | )2  |     | ( u−g |        |     |
Let us now discuss why performing such exact calculation is = −lnΣ − p −lnΣ − +C .
|                                                              |     |     |     |     |     |     |     |     | p   |     | Σ   | u   |     | Σ   | (7) |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| challengingforasimplebiologicalsystem.First,assoonasfunction |     |     |     |     |     |     |     | 2   |     |     |     |     |     |     |     |
|                                                              |     |     |     |     |     |     |     |     |     |     | p   |     |     | u   |     |
g relatingthevariablewewishtoinferwithobservationsisnon-
Weincorporatedtheconstanttermsinthe2ndlineaboveinto
linear,theposteriordistributionp (v|u ) maynottakeastandard φ
|           |         |     |                  |           |        |         | aconstantC.NowwecancomputethederivativeofF |     |     |     |     |     |     | over | :   |
| --------- | ------- | --- | ---------------- | --------- | ------ | ------- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | ---- | --- |
| shape—for | example |     | the distribution | in Fig. 1 | is not | normal. |                                            |     |     |     |     |     |     |      |     |
T h u s r e p r es e n tin g th e d i st r i bu ti o n p ( v | u ) re q u i r e s re p r es e n t in g ∂ F v −φ u− g (φ)
|     |     |     |     |     |     |     |     | = p | +   |     | g ′(φ). |     |     |     | (8) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
i n f in it e l y m a n y v alu e sp ( v | u ) fo r d if fe r e n t p o ss i b l e u r at h e r t h an ∂ φ Σ Σ
|     |     |     |     |     |     |     |     |     | p   | u   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|     |     |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |     |     | 201 |
| --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Fig.2. SolutionstoExercises2and3.Inpanelbwehavealsoincludedquantitiesthatwewillseelatercanberegardedaspredictionerrors.
|     |     |     |     |     |     |     |     | Theabovetermsarethepredictionerrors1: |     |     |     |     | ε   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Intheaboveequationweusedthechainruletocomputethe u expresseshow
|     |     | ′(φ) |     |     |     |     | φ   |     |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
secondterm,andg isaderivativeoffunctiongevaluatedat , muchthelightintensitydiffersfromthatexpectedifthesizeof
|     |     | ′(φ) |     | φ   |     |     | φ v |     |     | φ ε |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
soinourexampleg = 2 .Wecanfindourbestguess for thefooditemwas ,while denoteshowtheinferredsizediffers
|     |     |     | φ   |     |     |     |     |     |     |     | p   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
simplybychanging inproportiontothegradient: frompriorexpectations.Withthesenewvariablestheequationfor
|     |     |     |     |     |     |     |     | updating φ | simplifiesto: |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | --- | --- | --- | --- | --- |
∂
φ˙ = F .
|     |     |     |     |     |     |     | (9) | φ˙ =ε ′(φ)−ε |                | .   |     |           |         |     |          |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------------- | --- | --- | --------- | ------- | --- | -------- |
|     | ∂φ  |     |     |     |     |     |     | u g          | p              |     |     |           |         |     | (12)     |
|     |     |     | φ˙  |     |     | φ   |     | The neural   | implementation |     | of  | the model | assumes |     | that the |
In t he a b ov e eq u a t i o n i st h e r a t e o f c h a n g e o f w i th tim e. Le t v ,Σ ,andΣ
|     |     |     |     | φ   |     |     |     | modelparameters |     |     | areencodedinthestrengthsof |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | -------------------------- | --- | --- | --- | --- |
usn ot e th a t th e u p d a t e of i s v e r y i n t u it iv e . I t isd ri v en by tw o p p u
synapticconnections(astheyneedtobemaintainedovertheani-
| termsinEq.(8):thefirstmovesittowardsthemeanoftheprior, |     |     |     |     |     |     |     |                               |     |     | φ ε | ε    |                    |     |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ---- | ------------------ | --- | --- |
|                                                        |     |     |     |     |     |     |     | mal’slifetime),whilevariables |     |     | ,   | ,and | andthesensoryinput |     |     |
| thesecondmovesitaccordingtothesensorystimulus,andboth  |     |     |     |     |     |     |     |                               |     |     | u   |      | p                  |     |     |
uaremaintainedintheactivityofneuronsorneuronalpopulations
termsareweightedbythereliabilitiesofpriorandsensoryinput
(astheychangerapidlywhenthesensoryinputismodified).Inpar-
respectively.
ticular,wewillconsiderverysimpleneural‘‘nodes’’whichsimply
Nowpleasenotethattheaboveprocedureforfindingtheap-
changetheiractivityproportionallytotheinputtheyreceive,sofor
proximatedistributionofdistancetofooditemiscomputationally
example,Eq.(12)isimplementedinthemodelbyanodereceiving
muchsimplerthantheexactmethodpresentedatthestartofthe
inputequaltotherighthandsideofthisequation.Theprediction
paper.Togainmoreappreciationforthesimplicityofthiscompu-
errorscouldbecomputedbythenodeswiththefollowingdynam-
| tationwerecommenddoingthefollowingexercise. |     |     |     |     |     |     |     | ics2:   |     |     |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
|                                             |     |     |     |     |     |     |     | ε˙ =φ−v | −Σ  | ε   |     |     |     |     |     |
Exercise2. Write a computer program finding the most likely size (13)
|     |     |     |     |     |     |     |     | p   | p p | p   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ofthefooditem φ forthesituationdescribedinExercise1.Initialize ε˙ =u−g (φ)−Σ ε .
(14)
| φ = | v                                |     |     |     |     |                   |     | u   |     | u u |     |     |     |     |     |
| --- | -------------------------------- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | p ,andthenfinditsvaluesinthenext |     |     |     |     | 5timeunits(youcan |     |     |     |     |     |     |     |     |     |
φ( +(cid:49)t ) = φ( )+(cid:49)t ∂ /∂φ Itiseasytoshowthatthenodeswithdynamicsdescribedby
| useEuler’smethod,i.e.update |     |     |     | t   |     | t   | F with |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:49)t =0 . Eqs. (13)–(14) converge to the values defined in Eqs. (10)–(11).
|     | 01). |     |     |     |     |     |     |                                |     |     |     | ε˙            |     | ε˙  |        |
| --- | ---- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ------------- | --- | --- | ------ |
|     |      |     |     |     |     |     |     | OnceEqs.(13)–(14)converge,then |     |     |     | = 0,sosetting |     |     | = 0and |
ε
Fig. 2(a) shows a solution to Exercise 2. Please notice that it solvingEqs.(13)–(14)for ,oneobtainsEqs.(10)–(11).
rapidlyconvergestothevalueof φ ≈ 1 . 6,whichisalsothevalue ThearchitectureofthenetworkdescribedbyEqs.(12)–(14)is
that maximizes the exact posterior probability p (v|u ) shown in showninFig.3.Letusconsiderthecomputationsinitsnodes.The
|        |     |     |     |     |     |     |     | node ε receives                                      |     | excitatory | input from | node | φ , | inhibitory | input |
| ------ | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | ---------- | ---------- | ---- | --- | ---------- | ----- |
| Fig.1. |     |     |     |     |     |     |     | p                                                    |     |            |            |      |     |            |       |
|        |     |     |     |     |     |     |     | fromatonicallyactiveneuronviaaconnectionwithstrength |     |            |            |      |     |            | v ,   |
p
|     |     |     |     |     |     |     |     | and inhibitory | input | from | itself via | a connection |     | with | strength |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ----- | ---- | ---------- | ------------ | --- | ---- | -------- |
2.3. Apossibleneuralimplementation
|     |     |     |     |     |     |     |     | Σ , so it implements |     | Eq. (13). | The | nodes | φ and | ε analogously |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --------- | --- | ----- | ----- | ------------- | --- |
|     |     |     |     |     |     |     |     | p                    |     |           |     |       |       | u             |     |
implementEqs.(12)and(14),butheretheinformationexchange
Onecanenvisagemanypossiblewaysinwhichthecomputa-
betweenthemisadditionallyaffectedbyfunctiong,andwewill
| tion   | described | in previous |       | subsection | could        | be implemented | in     |              |       |         |           |         |      |         |     |
| ------ | --------- | ----------- | ----- | ---------- | ------------ | -------------- | ------ | ------------ | ----- | ------- | --------- | ------- | ---- | ------- | --- |
|        |           |             |       |            |              |                |        | discuss this | issue | in more | detail in | Section | 2.5. | We have | now |
| neural | circuits. | In this     | paper | we         | will present | a possible     | imple- |              |       |         |           |         |      |         |     |
describedallthedetailsnecessarytosimulatethemodel.
mentationwhichsatisfiestheconstraintsoflocalcomputationand
plasticitydescribedintheIntroduction.Itslightlydiffersfromthe
|     |     |     |     |     |     |     |     | Exercise3. | Simulate | the model | from | Fig. | 3 for | the | problem |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --------- | ---- | ---- | ----- | --- | ------- |
originalimplementationwhichiscontainedinAppendixA. φ = v , ε = ε =
|     |     |     |     |     |     |     |     | fromExercise1.Inparticular,initialize |     |     |     |     | p p | u   | 0,and |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | ----- |
Whilethinkingabouttheneuralimplementationoftheabove findtheirvaluesforthenext5unitsoftime.
computation,itishelpfultonotethattherearetwosimilarterms
inEq.(8),soletusdenotethembynewvariables.
φ−v
ε = p 1 Intheoriginalmodel(Friston,2005)thepredictionerrorswerenormalized
| p   | Σ   |     |     |     |     |     | (10) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
slightlydifferentlyasexplainedinAppendixA.
p
u−g (φ) 2 Theoriginalmodeldoesnotprovidedetailsonthedynamicsofthenodes
ε = . (11) computingpredictionerror,butweconsidersampledescriptionoftheirdynamics
u Σ
|     |     | u   |     |     |     |     |     | toillustratehowthesenodescanperformtheircomputation. |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |

| 202 |     |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |                                             |     |     |     |     |     |     |            |
| --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- |
|     |     |     |     |     |                                                         |     |     | theexpectedvalueovertrials.Thiswillhappenif |     |     |     |     |     | v   | = ⟨φ⟩,i.e. |
p
|     |     |     |     |     |     |     |     | when | v isindeedequaltotheexpectedvalueof |     |     |     |     | φ .Analogously, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------------------------------- | --- | --- | --- | --- | --------------- | --- |
p
|     |     |     |     |     |     |     |     | theexpectedvalueofchangeinΣ |     |     |     | is0when: |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | -------- | --- | --- | --- |
p
|     |     |     |     |     |     |     |     |    |     |    |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(φ−v )2
|     |     |     |     |     |     |     |     |                                         | p   | 1    | .   |     |           |        |           |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | ---- | --- | --- | --------- | ------ | --------- |
|     |     |     |     |     |     |     |     |                                         |     | − =0 |     |     |           |        | (18)      |
|     |     |     |     |     |     |     |     |                                         | Σ2  | Σ    |     |     |           |        |           |
|     |     |     |     |     |     |     |     |                                         | p   | p    |     |     |           |        |           |
|     |     |     |     |     |     |     |     | RearrangingtheaboveconditiononeobtainsΣ |     |      |     |     |           | =⟨(φ−v | )2⟩,      |
|     |     |     |     |     |     |     |     |                                         |     |      |     |     |           | p      | p         |
|     |     |     |     |     |     |     |     | thustheexpectedvalueofchangeinΣ         |     |      |     |     | is0,whenΣ |        | isequalto |
|     |     |     |     |     |     |     |     |                                         |     |      |     |     | p         |        | p         |
φ .AnanalogousanalysiscanbemadeforΣ
|     |     |     |     |     |     |     |     | thevarianceof |           |     |        |          |     |            | u .      |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --------- | --- | ------ | -------- | --- | ---------- | -------- |
|     |     |     |     |     |     |     |     | Eqs.          | (15)–(17) | for | update | of model |     | parameters | simplify |
Fig.3. Thearchitectureofthemodelperformingsimpleperceptualinference. significantlywhentheyarewrittenintermsofpredictionerrors
Circlesdenoteneural‘‘nodes’’,arrowsdenoteexcitatoryconnections,whilelines (TRYITYOURSELF):
endedwithcirclesdenoteinhibitoryconnections.Labelsabovetheconnections
| encodetheirstrength,andlackoflabelindicatesthestrengthof1.Rectangles |     |     |     |     |     |     |     | ∂ F |     |     |     |     |     |     |      |
| -------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
|                                                                      |     |     |     |     |     |     |     |     | =ε  |     |     |     |     |     | (19) |
| indicatethevaluesthatneedtobetransmittedviatheconnectionstheylabel.  |     |     |     |     |     |     |     | ∂v  | p   |     |     |     |     |     |      |
p
∂
SolutiontoExercise3isshowninFig.2(b).Themodelconverges F 1ε2−Σ−1
|                                                           |     |     |     |     |     |     |     |     | =   |     |     |     |     |     | (20) |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
|                                                           |     |     |     |     |     |     |     | ∂Σ  |     | p p |     |     |     |     |      |
| tothesamevalueasinFig.2(a),buttheconvergenceisjustslower, |     |     |     |     |     |     |     | p   | 2   |     |     |     |     |     |      |
| asthemodelnowincludesmultiplenodesconnectedbyexcitatory   |     |     |     |     |     |     |     | ∂   |     |     |     |     |     |     |      |
F = 1ε2−Σ−1.
and inhibitory connections and such networks have oscillatory (21)
|                                                           |     |     |     |     |     |     |     | ∂Σ  | 2   | u u |     |     |     |     |     |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tendencies,sotheseoscillationsneedtosettleforthenetworkto |     |     |     |     |     |     |     | u   |     |     |     |     |     |     |     |
converge. The above rules for update of parameters correspond to very
|     |     |     |     |     |     |     |     | simple | synaptic | plasticity       | mechanisms. |              | All | rules    | include only  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | -------- | ---------------- | ----------- | ------------ | --- | -------- | ------------- |
|     |     |     |     |     |     |     |     | values | that     | can be ‘‘known’’ | by          | the synapse, |     | i.e. the | activities of |
2.4. Learningmodelparameters
pre-synapticandpost-synapticneurons,andthestrengthsofthe
As our imaginary animal perceives food items through its synapseitself.Furthermore,therulesareHebbian,inthesensethat
lifetime,itmaywishtorefineitsexpectationabouttypicalsizes theydependontheproductsofactivityofpre-synapticandpost-
v
offooditemsdescribedbyparameters v andΣ ,andaboutthe synapticneurons.Forexample,thechangein p inEq.(19)isequal
|     |     |     |     |     | p p |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
amount of error it makes observing light intensity, described by totheproductofpre-synapticactivity(i.e.1)andthepost-synaptic
|     |     |     |     |     |     |     |     |     | ε   | Similarly,thechangesinΣ |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | --- | --- |
parameterΣ .Thusitmaywishtoupdatetheparameters v , Σ , activity . inEqs. (20)–(21)depend
|     | u   |     |     |     |     |     | p p |     | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andΣ ontheproductsofpre-synapticandpost-synapticactivities,both
| u   | aftereachstimulustograduallyrefinethem. |        |           |            |     |           |     |         | ε   |     |     |     |     |     |     |
| --- | --------------------------------------- | ------ | --------- | ---------- | --- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
|     |                                         |        |           |            |     |           |     | equalto | .   |     |     |     |     |     |     |
| We  | wish to                                 | choose | the model | parameters |     | for which | the |         |     |     |     |     |     |     |     |
TheplasticityrulesofEqs.(20)–(21)alsodependonthevalue
perceivedlightintensitiesuareleastsurprising,orinotherwords
ofsynapticweightsthemselves,astheyincludetermsΣ−1.Forthe
mostexpected.Thuswewishtochooseparametersthatmaximize
( ) ( ) simple case considered in this section, the synapse ‘‘has access’’
| p u .However,pleaserecallthatp                   |     |     |     | u isdescribedbyacomplicated |     |       |           |        |             |     |             |           |     |                |     |
| ------------------------------------------------ | --- | --- | --- | --------------------------- | --- | ----- | --------- | ------ | ----------- | --- | ----------- | --------- | --- | -------------- | --- |
|                                                  |     |     |     |                             |     |       |           | to the | information | on  | its weight. | Moreover, |     | the dependence | of  |
| integralofEq.(5),soitwouldbedifficulttomaximizep |     |     |     |                             |     | ( u ) | directly. |        |             |     |             |           |     |                |     |
Nevertheless,itissimpletomaximizearelatedquantityp ( u ,φ) , synapticplasticityoninitialweightshasbeenseenexperimentally
whichisthejointprobabilityofsensoryinputuandourinferred (Chen et al., 2013), so we feel it is plausible for the dependence
foodsize φ .Notethatp ( u ,φ) = p (φ) p ( u|φ) ,soF = lnp ( u ,φ) , predictedbythemodeltobepresentinrealsynapses.However,
( ,φ) when the model is scaled up to include multiple features and
thusmaximizationofp u canbeachievedbymaximizingF.A sensoryinputsinSection4.1,termsΣ−1
willturnintoamatrix
moreformalexplanationforwhytheparameterscanbeoptimized
inverse(inEqs.(48)–(49)),sotherequiredchangesineachweight
| bymaximizingF | willbeprovidedinSection3. |     |        |       |           |              |     |      |        |                |     |          |          |     |              |
| ------------- | ------------------------- | --- | ------ | ----- | --------- | ------------ | --- | ---- | ------ | -------------- | --- | -------- | -------- | --- | ------------ |
|               |                           |     |        |       |           |              |     | will | depend | on the weights |     | of other | synapses | in  | the network. |
| The model     | parameters                |     | can be | hence | optimized | by modifying |     |      |        |                |     |          |          |     |              |
Nevertheless,wewillshowinSection5howthisproblemcanbe
| them proportionally |     | to  | the gradient | of  | F. Starting |     | with the |     |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ------------ | --- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
overcome.
| expression | in Eq. | (7) it is | straightforward |     | to find | the derivatives |     |     |     |     |     |     |     |     |     |
| ---------- | ------ | --------- | --------------- | --- | ------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Finally,wewouldliketodiscussthelimitsonparametersΣ.
| ofF over | v , Σ andΣ | (TRYITYOURSELF): |     |     |     |     |     |     |     |     |     |     |     |     |     |
| -------- | ---------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
p p u Although in principle the variance of a random variable can be
| ∂   | φ −v |     |     |     |     |     |     |     |     | Σ = | Σ = |     |     |     |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
F p e q u a l t o 0 , if p 0 o r u 0 , th e n E q . ( 1 3 ) o r ( 1 4 ) w ou ld n o t
| =   |     |     |     |     |     |     | (15) |     |     | ε   | ε   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
∂ v Σ co n v e r g e b u t in ste ad p o r u wo u ld d i ve r g e t o p o s it i v e o rn eg at iv e
| p   | p   |     |     |     |     |     |     | infinity.Similarly,ifΣ |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
werecloseto0,theconvergencewouldbe
|     |    |     |    |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∂ F 1 (φ−v )2 1 veryslow.Topreventthisfromhappening,theminimumvalueof
| =   |     | p − |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∂Σ Σ2 Σ (16) 1isimposedbyFriston(2005)ontheestimatedvariance.3
2
| p   |        | p     | p   |     |     |     |      |      |                                         |     |     |     |     |     |     |
| --- | ------ | ----- | --- | --- | --- | --- | ---- | ---- | --------------------------------------- | --- | --- | --- | --- | --- | --- |
| ∂   | ( u−g | (φ))2 |    |     |     |     |      |      |                                         |     |     |     |     |     |     |
| F   | 1      |       | 1   | .   |     |     |      | 2.5. | Learningtherelationshipbetweenvariables |     |     |     |     |     |     |
| =   |        |       | −   |     |     |     | (17) |      |                                         |     |     |     |     |     |     |
| ∂Σ  | 2 Σ2   |       | Σ   |     |     |     |      |      |                                         |     |     |     |     |     |     |
| u   |        | u     | u   |     |     |     |      |      |                                         |     |     |     |     |     |     |
Sofarwehaveassumedforsimplicitythattherelationshipg
Letusnowprovideanintuitionforwhytheparameterupdate between the variable being inferred and the stimulus is known.
rules have their particular form. We note that since parameters However,ingeneralitmaynotbeknown,ormayneedtobetuned.
| are updated | after | observing | each | food | item, and | different | food |     |     |     |     |     |     |     |     |
| ----------- | ----- | --------- | ---- | ---- | --------- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
itemsobservedduringanimal’slifetimehavedifferentsizes,the
parameters never converge. Nevertheless it is useful to consider √
|     |     |     |     |     |     |     |     | 3 Intheoriginalmodel,variablesλ= |     |     |     | Σ−1weredefined,andthesevariables |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | -------------------------------- | --- | --- | --- |
thevaluesofparametersforwhichtheexpectedvalueofchange
|     |     |     |     |     |     |     |     | were | encoded | in the synaptic | connections. | Formally, |     | this constraint | is known |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------- | --------------- | ------------ | --------- | --- | --------------- | -------- |
is0,asthesearethevaluesinvicinityofwhichtheparametersare
asahyperprior.Thisisbecausethevarianceorprecisionparametersareoften
likelytobe.Forexample,accordingtoEq.(15),theexpectedvalue
referredtomathematicallyashyperparameters.Wheneverweplaceconstraintson
| ofchangein | v is0when⟨(φ |     | −v  | )/Σ | ⟩ = 0,where⟨⟩denotes |     |     |                                                |     |     |     |     |     |     |     |
| ---------- | ------------ | --- | --- | --- | -------------------- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|            | p            |     |     | p p |                      |     |     | hyperparameterswenecessarilyinvokehyperpriors. |     |     |     |     |     |     |     |

R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 203
Fig.4. Architecturesofmodelswithlinearandnonlinearfunctiong.Circlesandhexagonsdenotelinearandnonlinearnodesrespectively.Filledarrowsandlinesended
withcirclesdenoteexcitatoryandinhibitoryconnectionsrespectively,andanopenarrowdenotesamodulatoryinfluence.
(v,θ) ′
So we will now consider function g that also depends on non-lineartransformationh andmodulatinginputreceivedfrom
parameterwhichwedenoteby θ . node ε viaaconnectionwithweight θ (alternatively,thiscould
u
Wewillconsidertwospecialcasesoffunctiong (v,θ) ,where beimplementedwithinthenode φ bymakingitreacttoitsinput
θ
theparameter hasaclearbiologicalinterpretation.First,letus differentially depending on its level of activity). The details of
considerasimplecaseofalinearfunction:g (v,θ) = θv ,asthen the neural implementation of these non-linear transformations
the model has a straightforward neural implementation. In this depend on the form of function h, and would be an interesting
case,Eqs.(12)–(14)describingthemodelsimplifyto: directionofthefuturework.
θ
Wealsonotethattheupdateoftheparameter ,i.e.gradientof
φ˙ =θε u −ε p (22) F over θ becomes:
ε˙ =φ−v −Σ ε (23) ∂
p p p p F
ε˙ =u−θφ−Σ ε (24) ∂θ
=ε
u h
(φ).
(29)
u u u
In this model, nodes φ and ε simply communicate through This rule is Hebbian for the top connection labelled by θ in
connectionswithweight θ asshowninFig.4(a).Furthermore,we Fig.4(b),asitisaproductofactivityofthepre-synapticandpost-
canalsoderivetheruleforupdatingtheparameter θ byfindingthe synaptic nodes. It would be interesting to investigate how such
gradientofFover θ ,asnowfunctionginEq.(7)dependson θ (TRY a plasticity rule could be realized for the other connection with
θ ε φ
ITYOURSELF): the weight of (from node u to ). We just note that for this
connectiontherulealsosatisfiestheconstraintoflocalplasticity
∂
F =ε φ. (25) (stated in the Introduction), as φ fully determines h (φ) , so the
∂θ u
changeinweightisfullydeterminedbytheactivityofpre-synaptic
Please note that this rule is again Hebbian, as the synaptic andpost-synapticneurons.
θ
weights encoding are modified proportionally to the activities
ofpre-synapticandpost-synapticneurons(seeFig.4(a)). 3. Free-energy
Second,letusconsideracaseofanonlinearfunction4g (v,θ)=
θ (v) (v) v
h ,whereh isanonlinearfunctionthatjustdependson , In this section we discuss how the computations in the
asitresultsinonlyslightlymorecomplexneuralimplementation. model relate to a technique of statistical inference involving
Furthermore,thissituationisrelevanttotheexampleofthesimple minimizationoffree-energy.Therearethreereasonsfordescribing
animal considered at the start of this section, as the light is this relationship. First, it will provide more insight for why the
proportionaltothearea,buttheproportionalityconstantmaynot parameters can be optimized by maximization of F. Second,
beknown(thiscaseisalsorelevanttothenetworkthatwewill the concept of free-energy is critical for understanding of more
discussinSection4.1).Inthiscase,Eqs.(12)–(14)describingthe complex models (Friston et al., 2013), which not only estimate
modelbecome: the most likely values of variables, but their distribution. Third,
φ˙ =θε h ′(φ)−ε (26) thefree-energyisaveryinterestingconceptonitsown,andhas
u p
applicationsinmathematicalpsychology(Ostwald,Kirilina,Starke,
ε˙ =φ−v −Σ ε (27)
p p p p &Blankenburg,2014).
ε˙ =u−θ h (φ)−Σ ε . (28) We now come back to the example of an inference by a
u u u
simpleorganism,anddiscusshowtheexactinferencedescribed
A possible network implementing this model is illustrated in
in Section 2.1 can be approximated. As we noted in Section 2.1,
Fig.4(b),whichnowincludesnon-linearelements.Inparticular,
theposteriordistributionp
(v|u )
mayhaveacomplicatedshape,so
φ ε
the node sends to node its activity transformed by a non-
u wewillapproximateitwithanotherdistribution,whichwedenote
θ (φ)
linear function, i.e. h . One could imagine that this could be (v) (v)
q .Importantly,wewillassumethatq hasastandardshape,
φ
implementedbyanadditionalnodereceivinginputfromnode ,
sowewillbeabletocharacterizeitbyparametersofthistypical
transformingitviaanon-lineartransformationhandsendingits (v)
distribution.Forexample,ifweassumethatq isnormal,then
ε θ
outputtonode viaaconnectionwiththeweight .Analogously,
u tofullydescribeit,wecaninferjusttwonumbers:itsmeanand
ε φ θ ′(φ)
the input from node to node needs to be scaled by h .
u variance,insteadofinfinitelymanynumberspotentiallyrequired
Again one could imagine that this could be implemented by an
tocharacterizeadistributionofanarbitraryshape.
φ
additionalnodereceivinginputfromnode ,transformingitviaa
Forsimplicity,herewewilluseanevensimplershapeofthe
approximatedistribution,namelythedeltadistribution,whichhas
φ
allitsmasscumulatedinonepointwhichwedenoteby (i.e.the
φ
4 AlthoughthiscasehasnotbeendiscussedbyFriston(2005),itwasdiscussed deltadistributionisequalto0forallvaluesdifferentfrom ,but
byRaoandBallard(1999). itsintegralisequalto1).Thuswewilltrytoinferfromobservation

| 204 |     |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
justoneparameter φ whichwillcharacterizethemostlikelyvalue normalizationterm.Second,aswewillseelater,itwillallowusto
of v . naturallyintroducelearningabouttheparametersofthemodel.
|     |     |          |      |           |         |                 |     | Letusfirstnotethatbyassumingthatq |     |     |     |     |     | (v) isadeltadistribu- |     |
| --- | --- | -------- | ---- | --------- | ------- | --------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --------------------- | --- |
| We  | now | describe | what | criterion | we wish | our approximate |     |                                   |     |     |     |     |     |                       |     |
distributiontosatisfy.Wewillseektheapproximatedistribution tion,thenegativefreeenergysimplifiesto:
(v)
q which is as close as possible to the actual posterior  ( ,v)
|     |     | (v|u | )   |     |     |     |     |     | (v) | p u | v   |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
distribution p . Mathematically, the dissimilarity between F = q ln d
q (v)
twodistributionsinmeasuredbytheKullback–Leiblerdivergence
|            |     |     |     |     |     |     |     |     |    |       |     |    |       |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ----- | --- | --- |
| definedas: |     |     |     |     |     |     |     |     | (v) | ( ,v) | v−  | (v) | (v)   | v   |     |
|            |     |     |     |     |     |     |     | =   | q   | lnp u | d   | q   | lnq d |     |     |

q (v)
| KL(q | (v), p (v|u | ))= | q (v) | ln   | d v. |     | (30) |     |               |     |     |     |     |     |      |
| ---- | ----------- | --- | ----- | ---- | ---- | --- | ---- | --- | ------------- | --- | --- | --- | --- | --- | ---- |
|      |             |     |       | (v|u | )    |     |      | =   | lnp ( u ,φ)+C |     | .   |     |     |     | (34) |
|      |             |     |       | p    |      |     |      |     |               |     | 1   |     |     |     |      |
ForreadersnotfamiliarwithKullback–Leiblerdivergencewe Inthetransitionfromthefirsttothesecondlineaboveweused
|     |     |     |     |     |     |     |     |     |     |     |     | ( / | ) = lna−lnb.Inthetransition |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- |
wouldlikeclarifywhyitisameasureofdissimilaritybetweenthe thepropertyoflogarithmsln a b
distributions. Please note that if the two distributions q (v) and fromthesecondlinetothethirdlineweusedthepropertyofa
|     |     |     |     |     |     |     |     |     |     | δ( ) |     | φ   |     |     | ( ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
p (v|u ) wereidentical,theratioq (v)/ p (v|u ) wouldbeequalto1, delta function x with centre that for any function h x , the
|     |     |     |     |     |     |     |     |     | δ(  | ) ( ) |     | (φ) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
soitslogarithmwouldbeequalto0,andsothewholeexpression integralof x h x isequaltoh .Furthermore,sincethevalue
ofthesecondintegralinthesecondlineoftheaboveequationdoes
inEq.(30)wouldbe0.TheKullback–Leiblerdivergencealsohas
|     |     |     |     |     |     |     |     | notdependon |     | φ (soitwillcancelwhenwecomputethederivative |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------------------------------------- | --- | --- | --- | --- | --- |
apropertythatthemoredifferentthetwodistributionsare,the
|                                                        |           |         |         |        |                |              |            | over   | φ )wedenoteitbyaconstantC |       |            |       | .               |              |               |
| ------------------------------------------------------ | --------- | ------- | ------- | ------ | -------------- | ------------ | ---------- | ------ | ------------------------- | ----- | ---------- | ----- | --------------- | ------------ | ------------- |
| higheritsvalueis(seeOstwaldetal.(2014)formoredetails). |           |         |         |        |                |              |            |        |                           |       |            |       | 1               |              |               |
|                                                        |           |         |         |        |                |              |            | Now    | using                     | p ( u | ,φ) =      | p (φ) | p ( u|φ) ,      | and ignoring | constant      |
| Since                                                  | we        | assumed | above   | that   | our simplified | distribution | is         |        |                           |       |            |       |                 |              |               |
|                                                        |           |         |         |        |                |              |            | C , we | obtain                    | the   | expression | for   | F we introduced |              | previously in |
| a delta                                                | function, |         | we will | simply | seek the value | of           | its centre | 1      |                           |       |            |       |                 |              |               |
(v)
parameter φ which minimizes the Kullback–Leibler divergence Eq.(6).Thusfindingapproximatedeltadistributionq through
definedinEq.(30). minimization of free-energy is equivalent to the inference of
ItmayseemthattheminimizationofEq.(30)isstilldifficult, featuresinthemodeldescribedintheprevioussection.Itisworth
because to compute term p (v|u ) present in Eq. (30) from Bayes’ notingthatEq.(34)statesthatthebestcentreforourapproximate
distribution(i.e.ourbestguessforthesizeofthefooditem)isthe
theorem(Eq.(4))oneneedstocomputethedifficultnormalization v =φ ( ,φ)
|                                                |      |       |          |     |               |            |          | value |     | whichmaximizesthejointprobabilityp |     |             |                |     | u .          |
| ---------------------------------------------- | ---- | ----- | -------- | --- | ------------- | ---------- | -------- | ----- | --- | ---------------------------------- | --- | ----------- | -------------- | --- | ------------ |
| integral                                       | (Eq. | (5)). | However, | we  | will now show | that there | exists   |       |     |                                    |     |             |                |     |              |
|                                                |      |       |          |     |               | (v)        |          | We    | now | discuss                            | how | the concept | of free-energy |     | will help us |
| anotherwayoffindingtheapproximatedistributionq |      |       |          |     |               |            | thatdoes |       |     |                                    |     |             |                |     |              |
tounderstandwhytheparametersofthemodelcanbelearntby
| not | involve | the complicated |     | computation | of  | the normalization |     |              |     |       |        |              |     |         |              |
| --- | ------- | --------------- | --- | ----------- | --- | ----------------- | --- | ------------ | --- | ----- | ------ | ------------ | --- | ------- | ------------ |
|     |         |                 |     |             |     |                   |     | maximization |     | of F. | Recall | from Section | 2.4 | that we | wish to find |
integral.
parametersforwhichthesensoryobservationsareleastsurprising,
| Substitutingthedefinitionofconditionalprobabilityp |     |     |     |     |     |     | (v|u ) = |     |     |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
( )
p ( u ,v)/ p ( u ) intoEq.(30)weobtain: i.e.thosewhichmaximizep u .Toseetherelationshipbetween
( )
|     |     |     |     |       |         |     |     | maximizing |     | p u and | maximizing |     | F, we | note that | according to |
| --- | --- | --- | --- | ----- | ------- | --- | --- | ---------- | --- | ------- | ---------- | --- | ----- | --------- | ------------ |
|     |     |     |    | q (v) | p ( u ) |     |     |            | (   | )       |            |     |       |           |              |
KL(q (v), (v|u )) = (v) v Eq.(33),p u isrelatedtothenegativefree-energyinthefollowing
|     | p   |     | q   | ln  | ( ,v) d |     |     |                                                 |       |       |      |          |     |     |          |
| --- | --- | --- | --- | --- | ------- | --- | --- | ----------------------------------------------- | ----- | ----- | ---- | -------- | --- | --- | -------- |
|     |     |     |     | p   | u       |     |     | way:                                            |       |       |      |          |     |     |          |
|     |     |     |    |     | (v)     |     |     |                                                 | ( )=F | +KL(q | (v), | (v|u )). |     |     |          |
|     |     | =   | (v) | q   | v       |     |     | lnp                                             | u     |       | p    |          |     |     | (35)     |
|     |     |     | q   | ln  | d       |     |     |                                                 |       |       |      |          |     |     |          |
|     |     |     |     | p ( | u ,v)   |     |     |                                                 |       |       |      |          |     |     |          |
|     |     |     |     |     |         |     |     | SinceKullback–Leiblerdivergenceisnon-negative,F |       |       |      |          |     |     | isalower |

(v) v ( ) boundonlnp ( u ) ,thusbymaximizingF wemaximizethelower
|     |     |     | +   | q d | lnp u |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
( )
|     |     |     |     |     |     |     |     | boundonlnp |     | u .Soinsummary,bymaximizingF |     |     |     |     | wecanboth |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---------------------------- | --- | --- | --- | --- | --------- |
|     |     |     |    |     |     |     |     |            |     |                              |     |     | (v) |     |           |
q (v) find an approximate distribution q (as discussed earlier), and
|     |     | =   | (v) |     | v+lnp | ( ). |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ----- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
q ln ( ,v) d u (31) optimize model parameters. However, there is a twist here: we
|     |     |     |     | p   | u   |     |     |      |             |     |             |     | ( )    |               |          |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- | ----------- | --- | ------ | ------------- | -------- |
|     |     |     |     |     |     |     |     | wish | to maximize |     | the average |     | of p u | across trials | (or here |
In the transition from the second to the third line we used observationsofdifferentfooditems).Thusoneachtrialweneed
(v)
thefactthatq isaprobabilitydistributionsoitsintegralis1. tomodifythemodelparametersjustalittlebit(ratherthanuntil
The integral in the last line of the above equation is called free- minimumoffreeenergyisreachedaswasthecasefor φ ).
energy,andwewilldenoteitsnegativebyF,becausewewillshow
| below,                               | that | for certain | assumptions |     | the negative     | free-energy | is  |                                  |     |     |     |     |     |     |     |
| ------------------------------------ | ---- | ----------- | ----------- | --- | ---------------- | ----------- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                                      |      |             |             |     |                  |             |     | 4. Scalingupthemodelofperception |     |     |     |     |     |     |     |
| equal(moduloaconstant)tothefunctionF |      |             |             |     | wedefinedandused |             |     |                                  |     |     |     |     |     |     |     |
intheprevioussection: In this section we will show how the model scales up to the
 ( ,v) networksinferringmultiplefeaturesandinvolvinghierarchy.
p u
| F = | q (v) | ln  | d v. |     |     |     | (32) |     |     |     |     |     |     |     |     |
| --- | ----- | --- | ---- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
(v)
q
|     |     |     |     |     |     |     |     | 4.1. | Increasingthedimensionofsensoryinput |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------------------------------------ | --- | --- | --- | --- | --- | --- |
Intheaboveequationweusedthepropertyoflogarithmsthat
| −lna | / = | /   |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
b lnb a.So,thenegativefree-energyisrelatedtothe Themodelnaturallyscalesuptothecaseofmultiplesensory
Kullback–Leiblerdivergenceinthefollowingway: inputs from which we estimate multiple variables. Such scaled
KL(q (v), p (v|u ))=−F +lnp ( u ). (33) model could be used to describe information processing within
|     |     |     |     |     |     |     |     | a cortical | area | (e.g. | primary | visual | cortex) | which infers | multiple |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | ----- | ------- | ------ | ------- | ------------ | -------- |
|     |     |     |     | ( ) |     | φ   |     |            |      |       |         |        |         |              |          |
Nowpleasenotethatlnp u doesnotdependon (whichis features (e.g. edges at different position and orientation) on the
|     |     |     |     | (v) |     | φ   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
a parameter describing q ), so the value of that minimizes basis of multiple inputs (e.g. information from multiple retinal
|     |     |     |     | (v) | (v|u ) |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the distance between q and p is the same value as receptorspreprocessedbythethalamus).Thissectionshowsthat
that which maximizes F. Therefore instead of minimizing the whenthedimensionalityofinputsandfeaturesisincreased,the
Kullback–LeiblerdivergencewecanmaximizeF,andthiswillhave dynamics of nodes in the networks and synaptic plasticity are
twobenefits:first,aswealreadymentionedabove,F iseasierto described by the same rules as in Section 2, just generalized to
computeasitdoesnotinvolvethecomplicatedcomputationofthe multipledimensions.

|     |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |     |     | 205 |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Table1
Rulesforcomputationofderivatives.Adenotesasymmetricmatrix.
|     |     |     |     | Originalrule         |     |          | Generalizationtomatrices |                   |     |       |           |     |     |     |
| --- | --- | --- | --- | -------------------- | --- | -------- | ------------------------ | ----------------- | --- | ----- | --------- | --- | --- | --- |
|     |     |     |     | ∂ ax 2 =2ax          |     |          | ∂x T                     | Ax =2Ax           |     |       |           |     |     |     |
|     |     |     |     | ∂ x                  |     |          | ∂                        | x                 |     |       |           |     |     |     |
|     |     |     |     | ifz=f(y),y=g(x),then | ∂   | ∂ y ∂    | ifz=f(y),y=g(x),then     |                   |     | ∂     | ∂ y T ∂ |     |     |     |
|     |     |     |     |                      | ∂ z | = ∂ x∂ z |                          |                   |     | ∂ z = | ∂ ∂ z     |     |     |     |
|     |     |     |     |                      | x   | y        |                          |                   |     | x     | x y       |     |     |     |
|     |     |     |     | ∂ ln a = 1           |     |          | ∂l n                     | |A| =A−1          |     |       |           |     |     |     |
|     |     |     |     | ∂ a a                |     |          | ∂                        | A                 |     |       |           |     |     |     |
|     |     |     |     | ∂ x 2 x2             |     |          | ∂xT                      | −1x               |     |       |           |     |     |     |
|     |     |     |     | a =−                 |     |          |                          | A =−(A−1x)(A−1x)T |     |       |           |     |     |     |
|     |     |     |     | ∂ a a 2              |     |          |                          | ∂ A               |     |       |           |     |     |     |
The only complication in explaining this case lies in the symmetric,sowecanusethetoptworulesinTable1tocompute
necessitytousematrixnotation,soletusmakethisnotationvery thegradientofthenegativefreeenergy(TRYITYOURSELF):
| explicit: |     | we will denote | single | numbers in italic | (e.g. x), | column |     |     |     |     |     |     |     |     |
| --------- | --- | -------------- | ------ | ----------------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
(φ,(cid:50))T
vectorsbybar(e.g.x),andmatricesinbold(e.g.x).Soweassume ∂ F ∂ g
|     |     |     |     |     |     |     |     | =−(cid:54)−1(φ−v |     | )+  |     | (cid:54)−1( u−g | (φ,(cid:50))). | (38) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --------------- | -------------- | ---- |
theanimalhasobservedsensoryinputuandestimatesthemost ∂φ p p ∂φ u
| likelyvalues                        |     | φ ofvariables | v   | .Wefurtherassumethattheanimal |     |     |     |                                                       |     |     |     |     |     |     |
| ----------------------------------- | --- | ------------- | --- | ----------------------------- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                                     |     |               |     | v                             |     |     |     | Intheaboveequation,termsappearwhicharegeneralizations |     |     |     |     |     |     |
| haspriorexpectationthatthevariables |     |               |     | comefrommultivariate          |     |     |     |                                                       |     |     |     |     |     |     |
v andcovariancematrix(cid:54) ofthepredictionerrorswedefinedforthesimplemodels:
| normaldistributionwithmean |     |                  |     | p   |     | p ,i.e. |     |                 |     |     |     |     |     |      |
| -------------------------- | --- | ---------------- | --- | --- | --- | ------- | --- | --------------- | --- | --- | --- | --- | --- | ---- |
| (v)=f                      |     | (v;v ,(cid:54) ) |     |     |     |         |     |                 |     |     |     |     |     |      |
| p                          |     | p p where:       |     |     |     |         | ε   | =(cid:54)−1(φ−v |     | )   |     |     |     |      |
|                            |     |                  |     |     |     |         | p   | p               | p   |     |     |     |     | (39) |
|                            |     |                  |     |    |     |        |     |                 |     |     |     |     |     |      |
( x;µ,(cid:54))= 1 − 1 ( x−µ)T(cid:54)−1( x−µ) . ε =(cid:54)− 1( u−g (φ,(cid:50))). (40)
| f   |     | (             | exp |     |     | (36) | u   | u        |       |       |          |              |            |     |
| --- | --- | -------------- | --- | --- | --- | ---- | --- | -------- | ----- | ----- | -------- | ------------ | ---------- | --- |
|     |     | π )N|(cid:54)| |     | 2   |     |      |     |          |       |       |          |              |            |     |
|     |     | 2              |     |     |     |      |     | With the | error | terms | defined, | the equation | describing | the |
IntheaboveequationNdenotesthelengthofvectorx,and|(cid:54)| updateof φ becomes:
denotesthedeterminantofmatrix(cid:54).Analogously,theprobability
|     |     |     |     |     |     |     | ˙   |     | ∂ g (φ,(cid:50))T |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
ofobservingsensoryinputgiventhevaluesofvariablesisgivenby φ =−ε + ε .
|     |     |     |     |     |     |     |     | p   |     | u   |     |     |     | (41) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
p ( u|v) = f ( u;g (v,(cid:50)),(cid:54) ) ,where(cid:50)areparametersoffunction ∂φ
u
g.Wedenotetheseparametersbyamatrix(cid:50),aswewillconsider
|              |                |                 |          |             |            |           |                                         | The partial | derivative |     | term in the | above | equation                 | is a matrix |
| ------------ | -------------- | --------------- | -------- | ----------- | ---------- | --------- | --------------------------------------- | ----------- | ---------- | --- | ----------- | ----- | ------------------------ | ----------- |
| a            | generalization | of the          | function | g discussed | in Section | 2.5, i.e. |                                         |             |            |     |             |       |                          |             |
|              |                |                 |          |             |            |           | thatcontainsineachentrywithco-ordinates |             |            |     |             |       | (, i j ) thederivativeof |             |
| (v,(cid:50)) |                | = (cid:50)h (v) |          |             | (v)        |           |                                         |             |            |     |             |       |                          |             |
g ,whereeachelementiofvectorh depends elementiofvectorg (φ,(cid:50)) over φ .Toseehowtheaboveequation
|     | v   |     |     |     |     |     |     |     |     |     | j   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onlyon i .Thisfunctioncorrespondstoanassumptionoftenmade simplifiesforourchoiceoffunctiong,itishelpfulwithoutlossof
bymodelsoffeatureextraction(Bell&Sejnowski,1995;Olshausen generalitytoconsideracaseof2featuresbeingestimatedfrom2
&Field,1995),thatstimuliareformedbyalinearcombinationof
stimuli.Then:
features.5Moreover,suchafunctiongcanbeeasilycomputedasit
|     |     |     |     |     |     |     |     |     |     | θ  | ( φ ) + | θ ( φ )  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | --- |
i s e q u a l to a n i n p u t to a l a ye r o f ne u r o n s fro m anotherlayerwith (φ,(cid:50))=(cid:50)h (φ)= , h , h .
|     |     |     |     |     |     |     | g   |     |     | θ 1 | 1 ( φ 1 ) | θ 1 2 ( φ 2 ) |     | (42) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------- | --- | ---- |
a ct i v it y h (v ) v i a c o nn ec ti o n s w i th st r e n g th (cid:50) . , h + , h
|     |     |     |     |     |     |     |     |     |     | 2   | 1 1 | 2 2 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
Wecanstatethenegativefreeenergy,analogouslyasforthe
|     |     |     |     |     |     |     |     | Hence we | can | find the | derivatives | of elements | of  | the above |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | -------- | ----------- | ----------- | --- | --------- |
simplemodelconsideredinEq.(7)(TRYITYOURSELF):
|     |       |             |     |     |     |     | vectorovertheelementsofvector |                |       |           |             | φ : |     |      |
| --- | ----- | ----------- | --- | --- | --- | --- | ----------------------------- | -------------- | ----- | --------- | ----------- | --- | --- | ---- |
|     | =     | (φ ) + ( |φ | )   |     |     |     |                               |                |       |           |             |     |     |      |
| F   | l n p | ln p u      |     |     |     |     | ∂                             | ( φ ,(cid:50)) | θ    |           |             |    |     |      |
|     |       |             |     |     |     |     | g                             |                | ,     | h ′ ( φ ) | θ , h ′ ( φ | )   |     |      |
|     | 1     |             |     |     |     |     |                               |                | = 1 1 | 1         | 1 2         | 2 . |     | (43) |
= ( − ln | (cid:54) |− (φ − v )T(cid:54)− 1(φ−v ) ∂ φ θ h ′ ( φ ) θ h ′ ( φ )
|     |     | p   | p   | p p |     |     |     |     | 2 , 1 | 1   | 2 , 2 | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ----- | --- | --- | --- |
2
− ln|(cid:54) |−( u−g (φ,(cid:50)))T(cid:54)−1( u−g (φ,(cid:50))))+C . NowwecanseethatEq.(41)canbewrittenas:
|     |     | u   |     |     |     | (37) |     |        |                 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | ------ | --------------- | --- | --- | --- | --- | --- |
|     |     |     |     | u   |     |      | ˙   |        |                 |     |     |     |     |     |
|     |     |     |     |     |     |      | φ   | =−ε +h | ′(φ)×(cid:50)Tε |     | .   |     |     |     |
Analogouslyasbefore,tofindthevectorofmostlikelyvalues p u (44)
offeatures φ ,wewillcalculatethegradient(vectorofderivatives Intheaboveequation×denoteselementbyelementmultipli-
∂ F /∂φ )whichwewilldenoteby ∂ F /∂φ .Wewillusetheelegant ′(φ)×(cid:50)Tε
|     | i   |     |     |     |     |     | cation,sotermh |     |     |     | isavectorwhereitselementiisequal |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | -------------------------------- | --- | --- | --- |
property that rules for computation of derivatives generalize to ′(φ) andelementiofvector(cid:50)Tε u
|     |     |     |     |     |     |     | toaproductofh |     |     |     |     |     | .Analogously, |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | ------------- | --- |
vectors and matrices. To get an intuition for these rules we i u
asforthesimplemodel,predictionerrorscouldbecomputedby
recommend the following exercise that shows how the rule nodeswiththefollowingdynamics:
∂ x2/∂ x=2xgeneralizestovectors.
|     |           |                        |                 |                         |     |     | ε˙  | =φ−v         | −(cid:54)    | ε        |                |      |       |           |
| --- | --------- | ---------------------- | --------------- | ----------------------- | --- | --- | --- | ------------ | ------------ | -------- | -------------- | ---- | ----- | --------- |
|     |           |                        |                 |                         |     |     | p   |              | p p          | p        |                |      |       | (45)      |
| E x | er ci s e | 4 . S h o w t h at     | fo r a nyvector | xthegradientoffunctiony |     | =   | ε˙  |              |              |          |                |      |       |           |
|     |           |                        |                 |                         |     |     |     | =u−(cid:50)h | (φ)−(cid:54) | ε .      |                |      |       | (46)      |
| xT  | x is e q  | u a lt o : ∂ y / ∂ x = | 2 x .           |                         |     |     | u   |              |              | u u      |                |      |       |           |
|     |           |                        |                 |                         |     |     |     | It is easy   | to           | see that | Eqs. (45)–(46) | have | fixed | points at |
UsingananalogousmethodasthatinthesolutiontoExercise values given by Eqs. (39)–(40) by setting the left hand sides of
4 (at the end of the paper) one can see that several other rules Eqs. (45)–(46) to 0. The architecture of the network with the
generalizeassummarizedinTable1.Theserulescanbeappliedfor dynamicsdescribedbyEqs.(44)–(46)isshowninFig.5,anditis
symmetricmatrices,butsince(cid:54)arecovariancematrices,theyare
analogoustothatinFig.4(b).
Analogouslyasforthesimplemodel,onecanalsofindtherules
forupdatingparametersencodedinsynapticconnections,which
5 generalizetherulespresentedpreviously.Inparticular,usingthe
InthemodelofRaoandBallard(1999)thesparsecodingwasachievedthrough
introductionofadditionalpriorexpectationthatmostφ arecloseto0,butthe topformulainTable1itiseasytoseethat:
i
| sparsecodingcanalsobeachievedbychoosingashapeoffunctionhsuchthath(v |     |     |     |     |     | )   |     |     |     |     |     |     |     |     |
| ------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|                                                                     |     |     |     |     |     | i   | ∂   | F   |     |     |     |     |     |     |
aremostlycloseto0,butonlyoccasionallysignificantlydifferentfromzero(Friston, =ε . (47)
|     |     |     |     |     |     |     | ∂v  | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2008).
p

| 206 |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
togeneralizethemodelfrom2layerstomultiplelayers.Insuch
generalizedmodeltherulesdescribingdynamicsofneuronsand
plasticityofsynapsesremainexactlythesame,andonlynotation
|     |     |     |     |     |     | has to be | modified | to describe | presence of multiple |     | layers of |
| --- | --- | --- | --- | --- | --- | --------- | -------- | ----------- | -------------------- | --- | --------- |
hierarchy.
v
|     |     |     |     |     |     | Weassumethattheexpectedvalueofactivityinonelayer |     |     |     |     | i   |
| --- | --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- |
v
|     |     |     |     |     |     | dependsontheactivityinthenextlayer |                |     | i+1 : |     |      |
| --- | --- | --- | --- | --- | --- | ---------------------------------- | -------------- | --- | ----- | --- | ---- |
|     |     |     |     |     |     | E ( u )=g                          | (v ,(cid:50) ) |     |       |     | (51) |
|     |     |     |     |     |     |                                    | 1 2 1          |     |       |     |      |
|     |     |     |     |     |     | (v )=g                             | (v ,(cid:50) ) |     |       |     |      |
E
|     |     |     |     |     |     | 2   | 2 3 2 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
(v )=....
E 3
v
Fig.5. Thearchitectureofthemodelinferring2featuresfrom2sensorystimuli. Tosimplifythenotationwecoulddenoteuby ,andthenthe
1
Notation as in Fig. 4(b). To help identify which connections are intrinsic and likelihoodofactivityinlayeribecomes:
extrinsictoeachlevelofhierarchy,thenodesandtheirprojectionsineachlevelof
|     |     |     |     |     |     | p (v |v | )=f (v ;g (v | ,(cid:50) ),(cid:54) | ).  |     | (52) |
| --- | --- | --- | --- | --- | --- | ------- | ------------ | -------------------- | --- | --- | ---- |
hierarchyareshowningreen,blueandpurplerespectively(intheonlineversion). i i+1 i i i+1 i i
(Forinterpretationofthereferencestocolourinthisfigurelegend,thereaderis
Inthismodel,(cid:54)
referredtothewebversionofthisarticle.) i parametrizethecovariancebetweenfeatures
ineachlevel,and(cid:50)
i parametrizehowthemeanvalueoffeatures
UsingthetwobottomformulasinTable1onecanfindtherules inoneleveldependsonthenext.Letusassumethesameformof
|     |     |     |     |     |     |     |     | (v ,(cid:50) | ) = (cid:50) (v | )   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --------------- | --- | --- |
forupdateofcovariancematrices(TRYITYOURSELF): functiong asbefore,i.e.g i+1 h i+1 .Byanalogyto
|     |     |     |     |     |     |     |     | i   | i i |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
themodeldescribedintheprevioussubsection,onecanseethat
∂ F 1ε
= εT −(cid:54)−1 inferenceofthefeaturesinalllayersonthebasisofsensoryinput
|     | p   |     |     |     | (48) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
∂(cid:54) 2 p p canbeachievedinthenetworkshowninFig.6(a).Inthisnetwork
p
thedynamicsofthenodesaredescribedby:
∂ F 1ε
|           | = εT | −(cid:54)−1. |     |     |      |       |                    |         |     |     |      |
| --------- | ---- | ------------- | --- | --- | ---- | ----- | ------------------ | ------- | --- | --- | ---- |
| ∂(cid:54) | u    | u u           |     |     | (49) | ˙     |                    |         |     |     |      |
|           | 2    |               |     |     |      | φ =−ε | +h ′(φ ).∗(cid:50) | Tε      |     |     | (53) |
|           | u    |               |     |     |      | i     | i i                | i−1 i−1 |     |     |      |
rameters(cid:50)isabitmoretedious,
T he d e ri va t io n o f u p d at e o f p a ε˙ =φ −(cid:50) (φ )−(cid:54) ε .
|     |                |             |                |     |     |     | h i+1 |     |     |     | (54) |
| --- | -------------- | ----------- | -------------- | --- | --- | --- | ----- | --- | --- | --- | ---- |
| but | w e sh o w i n | A p p e n d | ix B t h a t : |     |     | i i | i     | i i |     |     |      |
Furthermore,byanalogytotheprevioussection,therulesfor
∂ F
|           | =ε (φ)T. |     |     |     |      | modifyingsynapticconnectionsinthemodelbecome: |     |     |     |     |     |
| --------- | -------- | --- | --- | --- | ---- | --------------------------------------------- | --- | --- | --- | --- | --- |
| ∂(cid:50) | u h      |     |     |     | (50) |                                               |     |     |     |     |     |
∂
TheaboveplasticityrulesofEqs.(47)–(50)areHebbianinthe F = 1εεT −(cid:54)−1
(55)
|                                                      |              |                 |                |                        |                      | ∂(cid:54) | 2 i i i  |     |     |     |      |
| ---------------------------------------------------- | ------------ | --------------- | -------------- | ---------------------- | -------------------- | --------- | -------- | --- | --- | --- | ---- |
| samesensetheywereforthesimplemodel—forexampleEq.(48) |              |                 |                |                        |                      | i         |          |     |     |     |      |
|                                                      | Σ            |                 |                |                        | ε ε                  | ∂         |          |     |     |     |      |
| i m p                                                | l ies t ha t | p ,i ,j s h o u | l d b e u pd a | t e d pr o p o rt i on | a ll y t o p ,i p ,j | , F       |          |     |     |     |      |
|                                                      |              |                 |                |                        |                      | =ε        | h (φ )T. |     |     |     | (56) |
i . e. t o th e p ro d u c t o f a c t ivi t y of p r e - syn a p ti c a n d p o st - sy n a p ti c ∂ (cid:50) i i+1
i
| neurons. | However, | the | rules of update | of covariance | matrices of |     |     |     |     |     |     |
| -------- | -------- | --- | --------------- | ------------- | ----------- | --- | --- | --- | --- | --- | --- |
Eqs.(48)–(49)containmatrixinverses(cid:54)−1.Thevalueofeachentry
|     |     |     |     |     |     | The hierarchical |     | structure of | the model in | Fig. 6(a) | parallels |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | ------------ | --------- | --------- |
inmatrixinversedependsonallmatrixelements,soitisdifficult thehierarchicalstructureofthecortex.Furthermore,itisworth
how it can be ‘‘known’’ by a synapse that encodes just a single notingthatdifferentlayerswithinthecortexcommunicatewith
element.Nevertheless,wewillshowinSection5howthemodel higher and lower sensory areas (as illustrated schematically in
canbeextendedtosatisfytheconstraintoflocalplasticity. Fig.6(b)),whichparallelthefactthatdifferentnodesinthemodel
communicatewithotherlevelsofhierarchy(Fig.6(a)).
4.2. Introducinghierarchy
5. Localplasticity
| Sensory | cortical | areas | are organized | hierarchically, | such that |     |     |     |     |     |     |
| ------- | -------- | ----- | ------------- | --------------- | --------- | --- | --- | --- | --- | --- | --- |
areasinlowerlevelsofhierarchy(e.g.primaryvisualcortex)infer Theplasticityrulesforsynapsesencodingmatrix(cid:54)(describing
presence of simple features of stimuli (e.g. edges), on the basis the variance and co-variance of features or sensory inputs)
of which the sensory areas in higher levels of hierarchy infer introduced in the previous section (Eqs. (48), (49) and (55))
presenceofmoreandmorecomplexfeatures.Itisstraightforward include terms equal to the matrix inverse (cid:54)−1. Computing each
Fig.6. (a)Thearchitectureofthemodelincludingmultiplelayers.Forsimplicityonlythefirsttwolayersareshown.NotationasinFig.5.(b)Extrinsicconnectivityofcortical
layers.

R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 207
Fig.7. Predictionerrornetworksthatcanlearntheuncertaintyparameterwith
localplasticity.NotationasinFig.4(b).(a)Singlenode.(b)Multiplenodesfor
multidimensionalfeatures.
Fig.8. ChangesinestimatedvarianceduringlearninginExercise5.
element of the inverse
(cid:54)−1
requires not only the knowledge of
Thusweseethatthepredictionerrornodehasafixedpointat
the corresponding element of (cid:54), but also of other elements. For
thedesiredvalue(cf.Eq.(57)).Letusnowconsiderthefollowing
example,inacaseof2-dimensionalvectoru,theupdaterulefor ruleforplasticityoftheconnectionencodingΣ:
the synaptic connection encoding Σ u,1,1 (Eq. (49)) requires the i
computationofΣ
u
−
,1
1
,1
=Σ
u,2,2
/|(cid:54)
u
|.Hencethechangeofsynaptic (cid:49)Σ
i
=α(ε
i
e
i
−1 ). (63)
weight Σ u,1,1 depends on the value of the weight Σ u,2,2 , but Accordingtothisruletheweightismodifiedproportionallyto
these are the weights of connections between different neurons
theproductofactivitiesofpre-synapticandpost-synapticneurons
(see Fig. 5), thus the update rule violates the principle of the α
decreasedbyaconstant,withalearningrate .Toanalysetowhat
local plasticity stated in the Introduction. Nevertheless, in this
values this rule converges, we note that the expected change is
section we show that by slightly modifying the architecture of
equalto0when:
thenetworkcomputingpredictionerrors,theneedforcomputing
matrixinversesintheplasticityrulesdisappears.Inotherwords, ⟨ε i e i −1⟩=0 . (64)
wepresentanextensionofthemodelfromtheprevioussectionin
SubstitutingEqs.(61)–(62)intotheaboveequationandrear-
whichlearningthevaluesofparametersΣsatisfiestheconstraint
rangingtermsweobtain:
of local plasticity. To make the description as easy to follow as
p
an
o
d
ss
s
ib
in
le
g
,
le
w
f
e
ea
s
t
t
u
ar
r
t
e
w
on
ith
ea
c
c
o
h
n
l
s
e
i
v
d
e
e
l
r
,
i
a
n
n
g
d
th
th
e
e
c
n
as
g
e
en
of
er
s
a
in
li
g
z
l
e
e
i
s
t
e
t
n
o
s
i
o
n
r
c
y
re
in
a
p
se
u
d
t ⟨(φ
i
−g
Σ i
(φ
i+1
))2⟩
=1 . (65)
dimensionofinputsandfeatures. i
SolvingtheaboveequationforΣ weobtainEq.(58).Thusin
i
summarythenetworkinFig.7(a)computesthepredictionerror
5.1. Learningvarianceofasinglepredictionerrornode
andlearnsthevarianceofthecorrespondingfeaturewithalocal
Hebbianplasticityrule.Togainmoreintuitionforhowthismodel
Instead of considering the whole model we now focus on
workswesuggestthefollowingexercise.
computationsinasinglenodecomputingpredictionerror.Inthe
modelwewishthepredictionerroroneachleveltoconvergeto:
Exercise5. SimulatelearningofvarianceΣ overtrials.Forsimplic-
i
ε i = φ i −g Σ i (φ i+1 ) . (57) i s t u y m ,o e n t l h y a s t im va u r l i a a t b e le t s he φ n a e r t e w c o o r n k st d a e n s t c . ri O b n ed ea b c y h E t q r s ia . l (5 g 9 en )– e ( r 6 a 0 te ), in an pu d t a φ s-
i i
from a normal distribution with mean 5 and variance 2, while set
the In m t e h a e n a p b re o d v i e ct e e q d u b at y io th n e Σ le i v is el t a h b e o v v a e r ) i : anceoffeature φ i (around g φ i (φ i+1 ) = 5(sothattheupperlevelcorrectlypredictsthemeanof
).Simulatethenetworkfor 20timeunits,andthenupdateweight
i
Σ i =⟨(φ i −g i (φ i+1 ))2⟩. (58) Σ i withlearningrate α =0 . 01.Simulate1000trialsandplothowΣ i
changesacrosstrials.
A sample architecture of the model that can achieve this
TheresultsofsimulationsareshowninFig.8,andtheyillustrate
computationwithlocalplasticityisshowninFig.7(a).Itincludes
thatthesynapticweightΣ approachesthevicinityofthevariance
anadditionalinhibitoryinter-neurone whichisconnectedtothe i
i φ
of .
predictionerrornode,andreceivesinputfromitviatheconnection i
ε
withweightencodingΣ.Thedynamicsofthismodelisdescribed It is also worth adding that i in the model described by
i
Eqs.(59)–(60)convergestothepredictionerror(Eq.(61)),when
bythefollowingsetofequations:
φ
oneassumesthat areconstantorchangeonmuchslowertime-
ε˙ i =φ i −g i (φ i+1 )−e i (59) scale than ε i and e i . This convergence takes place because the
e˙ =Σε −e . (60) fixedpointofthemodelisstable,whichcanbeshownusingthe
i i i i
standarddynamicalsystemstheory(Strogatz,1994).Inparticular,
Thelevelsofactivityatthefixedpointcanbefoundbysetting sinceEqs.(59)–(60)onlycontainlinearfunctionsofvariables ε and
i
thelefthandsidesofEqs.(59)–(60)to0andsolvingtheresulting e,theirsolutionhasaformofexponentialfunctionsoftimet,e.g.
i
setofsimultaneousequations(TRYITYOURSELF): ε( t )=cexp (λ t )+ε∗ ,wherecand λ areconstants,and ε∗ isthe
i i λ i
ε =
φ
i
−g
i
(φ
i+1
)
(61)
v
th
a
e
lu
f
e
ix
a
e
t
d
t
p
h
o
e
in
fi
t
x
:
e
w
d
h
p
e
o
n
in
λ
t.
<
Th
0
e
,t
s
h
ig
e
n
ex
o
p
f
on
d
e
e
n
t
t
e
ia
rm
lt
i
e
n
r
e
m
s
d
th
e
e
cr
s
e
t
a
a
s
b
e
i
s
lit
w
y
it
o
h
f
i Σ
i time,and ε convergestothefixedpoint,whileif λ > 0,thefixed
i
e i =φ i −g i (φ i+1 ). (62) pointisunstable.Thevaluesof λ areequaltotheeigenvaluesof

| 208 |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thematrixintheequationbelow(Strogatz,1994),whichrewrites
Eqs.(59)–(60)inavectorform:
| ε     |      | ε            | φ  | )     |        |              |     |     |     |     |     |     |     |
| ------- | ----- | --------------- | --- | ------ | ------ | ------------ | --- | --- | --- | --- | --- | --- | --- |
| ˙       | 0 − 1 |                 | −g  | (φ i+1 |        |              |     |     |     |     |     |     |     |
| i =     |       | i +             | i   | i      | .      | (66)         |     |     |     |     |     |     |     |
| e˙      | Σ − 1 | e               | 0   |        |        |              |     |     |     |     |     |     |     |
| i       | i     | i               |     |        |        |              |     |     |     |     |     |     |     |
| To show | that  | the eigenvalues |     | of the | matrix | in the above |     |     |     |     |     |     |     |
equationarenegativeweusethepropertythatsumofeigenvalues
isequaltothetraceandtheproducttothedeterminant.Thetrace
anddeterminantofthismatrixare−1andΣ,respectively.Since
i
thesumofeigenvaluesisnegativeandtheirproductpositive,both
eigenvaluesarenegative,sothesystemisstable.
5.2. Learningthecovariancematrix
| The model | described | in  | the previous | subsection |     | scales up to |     |     |     |     |     |     |     |
| --------- | --------- | --- | ------------ | ---------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
largerdimensionoffeaturesandsensoryinputs.Thearchitecture
ofthescaledupnetworkisshowninFig.7(b),anditsdynamicsis
describedbythefollowingequations: Fig.9. Anexampleofatexture.
| ε˙ =φ | (φ  | )−e |     |     |     |     |     |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
−g (67) stimulus,thesmallerthecorrespondingpredictionerror,andthus
| i i | i i+1 | i   |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
˙ =(cid:54)ε −e . loweritsinfluenceonactivityonotherneuronsinthenetwork.
| e i i | i i |     |     |     |     | (68) |     |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
Second,themodelcanlearnpropertiesoffeaturesencodedin
Analogouslyasbefore,wecanfindthefixedpointbysettingthe covarianceofsensoryinput.Anexampleofsuchfeatureistexture,
lefthandsideoftheequationto0: which can be efficiently recognized on the basis of covariance,
irrespectivelyoftranslation(Harwood,Ojala,Pietikäinen,Kelman,
| ε =(cid:54) −1(φ | −g  | (φ )) |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i i i i+1 (69) &Davis,1995).Togetanintuitionforthisproperty,letusconsider
i
=φ (φ ). a n e x a m p l e o f c h e c k e r -b o a rd t e x tu r e ( F i g . 9 ). P l e a s e n o te th a t
| e −g |     |     |     |     |     | (70) |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
i i i i+1 a dja c e n t n o d e s h a v e a l w a y s op p o s it e c o l o u r – c o r r e s po n d ing t o
Thuswecanseethatnodes ε havefixedpointsatthevalues negative covariance, while the diagonal nodes have the same
equaltothepredictionerrors.Wecannowconsideralearningrule
colour–correspondingtopositivecovariance.
analogoustothatintheprevioussubsection: Third,theattentionalmodulationcanbeeasilyimplementedin
themodelbychangingthevarianceassociatedwiththeattended
| (cid:49)(cid:54) =α(ε | eT −1 | ).  |     |     |     | (71) |     |     |     |     |     |     |     |
| --------------------- | ----- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
i i i features (Feldman & Friston, 2010). Thus for example, attending
|         |            |     |          |          |           |          | to feature | i at level j | of the hierarchy |     | can be | implemented | by  |
| ------- | ---------- | --- | -------- | -------- | --------- | -------- | ---------- | ------------ | ---------------- | --- | ------ | ----------- | --- |
| To find | the values | to  | vicinity | of which | the above | rule may |            |              |                  |     |        |             |     |
converge,wecanfindthevalueof(cid:54) decreasingsynapticweightΣ j,i,i ,orinhibitingnodee j,i incaseof
i forwhichtheexpectedvalue
themodeldescribedinSection5,whichwillresultinalargereffect
oftherighthandsideoftheaboveequationisequalto0:
ofthenodeencodingthisfeatureontheactivityintherestofthe
| ⟨ε eT −1⟩=0 | .   |     |     |     |     |      | network. |          |          |             |     |              |     |
| ----------- | --- | --- | --- | --- | --- | ---- | -------- | -------- | -------- | ----------- | --- | ------------ | --- |
| i i         |     |     |     |     |     | (72) |          |          |          |             |     |              |     |
|             |     |     |     |     |     |      | In this  | paper we | included | description | of  | the modified | or  |
SubstitutingEqs.(69)–(70)intotheaboveequation,andsolving
for(cid:54) extended version of the model with local computation and
weobtain(TRYITYOURSELF):
i plasticity to better illustrate how computation proposed by the
(cid:54) =⟨(φ −g (φ ))(φ −g (φ ))T⟩. f re e - e n e r g y fr a m e w o rk c a n b e im p le m e n t e d i n n e u r a l c ir c u it s .
| i   | i i | i+1 i | i i+1 |     |     | (73) |               |                      |              |              |             |             |                  |
| --- | --- | ----- | ----- | --- | --- | ---- | ------------- | -------------------- | ------------ | ------------ | ----------- | ----------- | ---------------- |
|     |     |       |       |     |     |      | H o w e v e r | , i tw i ll b e n ec | ess a r y in | th e fu tu r | e t o n u m | e ri c a ll | y e v a lu a t e |
Wecanseethatthelearningrulehasastochasticfixedpoint the efficiency of learning in the proposed model and the free-
atthevaluescorrespondingtothecovariancematrix.Insummary, energyframeworkingeneral.Existingmodelsoffeatureextraction
thenodesinnetworkdescribedinthissectionhavefixedpointsat (Bell&Sejnowski,1997;Bogacz,Brown,&Giraud-Carrier,2001;
predictionerrorsandcanlearnthecovarianceofthecorresponding Olshausen & Field, 1995) and predictive coding (Rao & Ballard,
features,thustheproposednetworkmaysubstitutetheprediction
1999)havebeenshowntobeabletofindfeaturesefficientlyand
| error nodes | in the | model | shown | in Fig. 6, | and the | computation |           |               |           |         |        |         |        |
| ----------- | ------ | ----- | ----- | ---------- | ------- | ----------- | --------- | ------------- | --------- | ------- | ------ | ------- | ------ |
|             |        |       |       |            |         |             | reproduce | the receptive | fields of | neurons | in the | primary | visual |
willremainthesame.Butimportantlyintheproposednetworkthe
cortexwhentrainedwithnaturalimages.Itwouldbeinteresting
covarianceislearntwithlocalplasticityinvolvingsimpleHebbian to explicitly test in simulations if the model based on the free-
learning. energy framework can equally efficiently extract features from
naturalstimuliandadditionallylearnthevarianceandcovariance
| 6. Discussion |     |     |     |     |     |     | offeatures.     |                   |        |         |              |      |            |
| ------------- | --- | --- | --- | --- | --- | --- | --------------- | ----------------- | ------ | ------- | ------------ | ---- | ---------- |
|               |     |     |     |     |     |     | We have         | also demonstrated |        | that if | the dynamics |      | within the |
|               |     |     |     |     |     |     | nodes computing | prediction        | errors | takes   | place        | on a | time-scale |
Inthispaperwepresentedthemodelofperceptionandlearning
inneuralcircuitsbasedonthefree-energyframework.Thismodel much faster than in the whole network, these nodes converge
extends the predictive coding model (Rao & Ballard, 1999) in to stable fixed points. It is also worth noting that under the
φ
that it represents and learns not only mean values of stimuli or assumptionofseparationoftimescales,thenodescomputing
φ
features,but alsotheir variances,whichgives themodel several alsoconvergetoastablefixedpoint,becausevariables converge
newcomputationalcapabilities,aswenowdiscuss. to the values that maximize function F. It would be interesting
First,themodelcanweightincomingsensoryinformationby toinvestigatehowtoensurethatthemodelconvergestodesired
their reliability. This property arises in the model, because the values(ratherthanengagingintooscillatorybehaviour)alsowhen
predictionerrorsarenormalizedbydividingthembythevariance oneconsidersamorerealisticcaseoftime-scalesnotbeingfully
| of noise. | Thus the | more | noisy is | a particular | dimension | of the | separated. |     |     |     |     |     |     |
| --------- | -------- | ---- | -------- | ------------ | --------- | ------ | ---------- | --- | --- | --- | --- | --- | --- |

|     |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     |     |     |     | 209 |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Insummary,inthispaperwepresentedthefree-energytheory,
| which offers | a powerful | framework |     | for describing |     | computations |     |     |     |     |     |     |     |     |     |
| ------------ | ---------- | --------- | --- | -------------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
performedbythebrainduringperceptionandlearning.Theappeal
| of the similarity | in the       | organization  |           | of networks |            | suggested | by       |     |     |     |     |     |     |     |     |
| ----------------- | ------------ | ------------- | --------- | ----------- | ---------- | --------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
| this theory       | and observed | in            | the brain | invites     | attempts   |           | to map   |     |     |     |     |     |     |     |     |
| the currently     | relatively   | abstract      | models    |             | on details | of        | cortical |     |     |     |     |     |     |     |     |
| micro-circuitry,  | i.e. to      | map different |           | elements    | of         | the model | on       |     |     |     |     |     |     |     |     |
differentneuralpopulationswithinthecortex.Forexample,Bastos
| et al. (2012) | compared | a more | recent | version |     | of the | model |     |     |     |     |     |     |     |     |
| ------------- | -------- | ------ | ------ | ------- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
(Friston,2008)withthedetailsofthecorticalorganization.Such
comparisonsofthemodelswithbiologicalcircuitsarelikelytolead
toiterativerefinementofthemodels.
|      |                    |     |           |      |          |     |          | Fig.10. | Thearchitecturesoftheoriginalmodelperformingsimpleperceptual |     |     |     |     |     |     |
| ---- | ------------------ | --- | --------- | ---- | -------- | --- | -------- | ------- | ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
| Even | if the free-energy |     | framework | does | describe |     | cortical |         |                                                              |     |     |     |     |     |     |
inference.NotationasinFig.3.
| computation, | the mapping | between |     | the variables |     | in the | model |     |     |     |     |     |     |     |     |
| ------------ | ----------- | ------- | --- | ------------- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
andtheelementsofneuralcircuitmaynotbe‘‘clean’’butrather Analogouslyasbefore,wecanfindtherulesdescribingsynaptic
‘‘messy’’i.e.eachmodelvariableorparametermayberepresented plasticityinthemodel,bycalculatingthederivativesofF(Eq.(76))
|     |     |     |     |     |     |     |     | v   | σ   | σ   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bymultipleneuronsorsynapses.Theparticularimplementation over , and (TRYITYOURSELF):
|     |     |     |     |     |     |     |     |     | p p | u   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
oftheframeworkinthecorticalcircuitmaybeinfluencedbyother
∂
| constraintstheevolutionarypressureoptimizessuchasrobustness |     |     |     |     |     |     |     | F =ξ | σ−1 |     |     |     |     |     |     |
| ----------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
(80)
| todamage,energyefficiency,speedofprocessing,etc.Inanycase, |     |     |     |     |     |     |     | ∂v  | p p |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
p
| the comparison | of predictions |     | of theoretical |     | framework |     | like the |     |     |     |     |     |     |     |     |
| -------------- | -------------- | --- | -------------- | --- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
∂ F
free-energywithexperimentaldataoffershopeforunderstanding =ξ2−1 σ−1
|     |     |     |     |     |     |     |     | ∂σ  | p   | p   |     |     |     |     | (81) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
thecorticalmicro-circuits.
p
∂
|                 |     |     |     |     |     |     |     | F =ξ |     | σ  |     |     |     |     |      |
| --------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | ---- |
|                 |     |     |     |     |     |     |     |       | 2−1 | −1. |     |     |     |     | (82) |
| Acknowledgments |     |     |     |     |     |     |     | ∂σ    | u   | u   |     |     |     |     |      |
u
ThisworkwassupportedbyMedicalResearchCouncilgrantMC Theoriginalmodeldoesnotsatisfytheconstraintoflocalcom-
φ
UU 12024/5. The author thanks Karl Frison, John-Stuart Brittain, putationstatedintheIntroduction,becausethenodecomputing
DanielaMassiceti,LinusSchumacherandRuiCostaforreadingthe receivestheinputfrompredictionerrornodesscaledbyparame-
|     |     |     |     |     |     |     |     | σ   |     |     |     |     | σ   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
previous version of the manuscript and very useful suggestions, ters (seeFig.10),buttheparameters arenotencodedinthe
φ
andChrisMathys,PeterDayanandDiegoVidaurrefordiscussion. connectionsbetweennode andthepredictionerrornodes,but
|     |     |     |     |     |     |     |     | instead | in the | connections | among | the | prediction | error | neurons. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ----------- | ----- | --- | ---------- | ----- | -------- |
Nevertheless,wehaveshowninSection2.3thatbyjustchanging
| AppendixA. | Theoriginalneuralimplementation |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thewayinwhichpredictionerrorsarenormalizedthecomputa-
tioninthemodelbecomeslocal.
Intheoriginalmodel(Friston,2005),thepredictionerrorswere
definedinaslightlydifferentway:
|     |     |     |     |     |     |     |     | AppendixB. |     | Derivation | of plasticity |     | rule for | connections | be- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---------- | ------------- | --- | -------- | ----------- | --- |
φ−v
| ξ = | p   |     |     |     |     |     |      | tweenlayers |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| p   |     |     |     |     |     |     | (74) |             |     |     |     |     |     |     |     |
σ
|     | p   |     |     |     |     |     |     |                                                     |          |         |     |          |        | (cid:50) |          |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | -------- | ------- | --- | -------- | ------ | -------- | -------- |
|     |     |     |     |     |     |     |     | This                                                | Appendix | derives | the | rule for | update | of       | given in |
| u−g | (φ) |     |     |     |     |     |     |                                                     |          |         |     |          |        |          |          |
| ξ = | .   |     |     |     |     |     |     | Eq.(50).InordertousethetwotopformulasinTable1wehave |          |         |     |          |        |          |          |
u σ (75) toreshapethematrix(cid:50)intoavector.Toavoiddeathbynotation,
|     | u   |     |     |     | √   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Σ withoutlossofgenerality,letusconsiderthecaseof2dimensional
| Intheaboveequations, |     | σ   | =   | ,and | σ = | Σ ,i.e. | σ and |     |     |     |     |     |     |     |     |
| -------------------- | --- | --- | --- | ---- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
p p u u p s ti mu l i a n d f e at u r e s . S o l et us d e f in e th e v ec t o r o f p a r am e te rs :
| σ d e n o t | e t h e st a nd a r d | d e v i a t | i o n s o f | d is t r ib | u t i o n s | p ( v ) an | d p ( u| v ) |     |     |     |     |     |     |     |     |
| ----------- | --------------------- | ----------- | ----------- | ----------- | ----------- | ---------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
u θ = [ θ 1, , θ , , θ 2, , θ ,2 ]. N ow ,u s i n g tw o to p f o r m u la s i n Ta b le 1
r e s p e ct i v el y . W i th t h e p r e d i c t i o n e r ro r t e r m s d e f in e d in th i s w a y , 1 1 2 1 2
onecanfindthat:
thenegativefreeenergycomputedinEq.(7)canbewrittenas:
(φ,(cid:50))T
|        |            |     |       |     |     |     |      | ∂ F | ∂ g |     |     |     |     |     |      |
| ------ | ---------- | --- | ----- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | ---- |
|        | 1          |     | 1     |     |     |     |      | =   |     | ε . |     |     |     |     | (83) |
| F =−ln | σ − ξ 2−ln | σ − | ξ 2+C | .   |     |     | (76) | ∂θ  | ∂θ  | u   |     |     |     |     |      |
|        | p p        | u   | u     | 2   |     |     |      |     |     |     |     |     |     |     |      |
|        | 2          |     | 2     |     |     |     |      |     |     |     |     |     |     |     |      |
Thedynamicsofvariable φ isproportionaltothederivativeof FromEq.(42)wefind:
| t he ab o ve | eq u ati o n over | φ : |     |     |     |     |      |                                            |            |        |      |      |      |     |      |
| ------------ | ----------------- | --- | --- | --- | --- | --- | ---- | ------------------------------------------ | ---------- | ------ | ---- | ---- | ---- | --- | ---- |
|              |                   |     |     |     |     |     |      | ∂ g (φ ,                                   | (cid:50))T |  (φ ) | (φ ) |      |     |     |      |
|              |                   |     |     |     |     |     |      |                                            | =          | h 1    | h 2  | 0    | 0    | .   |      |
| ξ            | ′(φ ) ε           |     |     |     |     |     |      |                                            |            |        |      | (φ ) | (φ ) |     | (84) |
| φ˙ u g       | p .               |     |     |     |     |     |      | ∂ θ                                        |            | 0      | 0    | h 1  | h 2  |     |      |
| =            | −                 |     |     |     |     |     | (77) |                                            |            |        |      |      |      |     |      |
| σ            | σ                 |     |     |     |     |     |      |                                            |            |        |      |      |      |     |      |
|              | u p               |     |     |     |     |     |      | WecannowevaluatetherighthandsideofEq.(83): |            |        |      |      |      |     |      |
AnalogouslyasinSection2.3,thepredictionerrorsdefinedin  (φ )   (φ )ε 
|     |     |     |     |     |     |     |     |     |     | h 1 | 0   |     |     | h 1 u , 1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
E q s. ( 74 ) – (75)couldbecomputedinthenodeswiththefollowing ∂ (φ , (cid:50))T (φ ) ε  ( φ ) ε
|             |       |     |     |     |     |     |      | g   | ε   | = h 2 | 0   |  u , 1 | =  | h 2 u , 1  | .    |
| ----------- | ----- | --- | --- | --- | --- | --- | ---- | --- | --- | ------ | --- | ------- | --- | ----------- | ---- |
| d yn a m ic | s :   |     |     |     |     |     |      |     |     |        | (φ  | ) ε     |     | ( φ ) ε     | (85) |
|             |       |     |     |     |     |     |      | ∂ θ | u   |  0    | h   |  ,     |    | h ,        |      |
|             |       |     |     |     |     |     |      |     |     |        | (φ  | 1 ) u 2 |     | (φ 1 )ε u 2 |      |
| ξ˙ =φ−v     | −σ ξ  |     |     |     |     |     | (78) |     |     | 0      | h   |         |     | h u,2       |      |
| p           | p p p |     |     |     |     |     |      |     |     |        |     | 2       |     | 2           |      |
ξ˙ =u−g (φ)−σ ξ . Reshaping the right hand side of the above equation into a
(79)
| u   | u u |     |     |     |     |     |     | matrix,wecanseehowitcanbedecomposedintotheproductof |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
The architecture of the model described by Eqs. (77)–(79) is vectorsinEq.(50):
| shown in | Fig. 10. It is | similar | to that | in Fig. | 3, but | differs | in the |     |     |     |     |     |     |     |     |
| -------- | -------------- | ------- | ------- | ------- | ------ | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
|          |                |         |         |         |        |         |        |    |     |     |   |    |     |     |     |
i nf o r m a t i o n r e c e iv e d b y n o d e φ (w e w i l ldiscussthisdifferencein h ( φ ) ε , h ( φ ) ε , ε ,  ).
|     |     |     |     |     |     |     |     | 1   | u 1 | 2 u 1 | = u | 1 h (φ | ) h (φ |     | (86) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ------ | ------ | --- | ---- |
m o r e d e t a i la t t h e e n d o f t h is A pp en d ix ) . h ( φ ) ε , h ( φ ) ε , ε , 1 2
|     |     |     |     |     |     |     |     | 1   | u 2 | 2 u 2 | u   | 2   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |

| 210 |     |     |     | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |
| --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- |
Solutionstoexercises
Exercise1
| function | exercise1 |          |                    |            |                 |      |          |
| -------- | --------- | -------- | ------------------ | ---------- | --------------- | ---- | -------- |
| v_p = 3; | %         | mean of  | prior distribution |            | of food         | size |          |
| sigma_p  | = 1; %    | standard | deviation          | of prior   | = sqrt(1)       |      |          |
| sigma_u  | = 1; %    | standard | deviation          | of sensory | noise           | =    | sqrt(1)  |
| u = 2;   | %         | observed | light intensity    |            |                 |      |          |
| MINV =   | 0.01; %   | minimum  | value of           | v for      | which posterior |      | computed |
DV = 0.01; % interval between values of v for which posterior found
| MAXV = | 5; %              | maximum | value of | v for | which posterior |     | computed |
| ------ | ----------------- | ------- | -------- | ----- | --------------- | --- | -------- |
| vrange | = [MINV:DV:MAXV]; |         |          |       |                 |     |          |
numerator = normpdf (vrange,v_p,sigma_p) .* normpdf (u,vrange.^2,sigma_u);
| normalization | = sum            | (numerator | * DV); |     |     |     |     |
| ------------- | ---------------- | ---------- | ------ | --- | --- | --- | --- |
| p = numerator | / normalization; |            |        |     |     |     |     |
| plot (vrange, | p, ’k’);         |            |        |     |     |     |     |
xlabel (’v’);
ylabel (’p(v|u)’);
Exercise2
| function   | exercise2 |              |                    |              |         |      |     |
| ---------- | --------- | ------------ | ------------------ | ------------ | ------- | ---- | --- |
| v_p = 3;   | %         | mean of      | prior distribution |              | of food | size |     |
| Sigma_p    | = 1; %    | variance     | of prior           | distribution |         |      |     |
| Sigma_u    | = 1; %    | variance     | of sensory         | noise        |         |      |     |
| u = 2;     | %         | observed     | light intensity    |              |         |      |     |
| DT = 0.01; | %         | integration  | step               |              |         |      |     |
| MAXT =     | 5; %      | maximum      | time considered    |              |         |      |     |
| phi(1)     | = v_p; %  | initializing | the                | best guess   | of food | size |     |
for i = 2:MAXT/DT
| phi(i) | = phi(i-1)             | + DT | * ((v_p | - phi(i-1))/Sigma_p |                | +   | ... |
| ------ | ---------------------- | ---- | ------- | ------------------- | -------------- | --- | --- |
|        | (u-phi(i-1)^2)/Sigma_u |      |         | *                   | (2*phi(i-1))); |     |     |
end
| plot ([DT:DT:MAXT], |     | phi, ’k’); |     |     |     |     |     |
| ------------------- | --- | ---------- | --- | --- | --- | --- | --- |
xlabel (’Time’);
ylabel (’\phi’);
| axis ([0 | MAXT -2 3.5]); |     |     |     |     |     |     |
| -------- | -------------- | --- | --- | --- | --- | --- | --- |
Exercise3
| function   | exercise3 |              |                    |              |         |            |       |
| ---------- | --------- | ------------ | ------------------ | ------------ | ------- | ---------- | ----- |
| v_p = 3;   | %         | mean of      | prior distribution |              | of food | size       |       |
| Sigma_p    | = 1; %    | variance     | of prior           | distribution |         |            |       |
| Sigma_u    | = 1; %    | variance     | of sensory         | noise        |         |            |       |
| u = 2;     | %         | observed     | light intensity    |              |         |            |       |
| DT = 0.01; | %         | integration  | step               |              |         |            |       |
| MAXT =     | 5; %      | maximum      | time considered    |              |         |            |       |
| phi(1)     | = v_p; %  | initializing | the                | best guess   | of food | size       |       |
| error_p(1) | = 0; %    | initializing | the                | prediction   | error   | of food    | size  |
| error_u(1) | = 0; %    | initializing | the                | prediction   | error   | of sensory | input |
for i = 2:MAXT/DT
phi(i) = phi(i-1) + DT * (- error_p(i-1) + error_u(i-1) * (2*phi(i-1)));
error_p(i) = error_p(i-1) + DT * (phi(i-1) - v_p - Sigma_p * error_p(i-1));
error_u(i) = error_u(i-1) + DT * (u - phi(i-1)^2 - Sigma_u * error_u(i-1));
end
| plot ([DT:DT:MAXT], |     | phi, ’k’); |     |     |     |     |     |
| ------------------- | --- | ---------- | --- | --- | --- | --- | --- |
hold on

|      |                |     |          |         | R.Bogacz/JournalofMathematicalPsychology76(2017)198–211 |     |     |     |     |     |     |     | 211 |
| ---- | -------------- | --- | -------- | ------- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| plot | ([DT:DT:MAXT], |     | error_p, | ’k--’); |                                                         |     |     |     |     |     |     |     |     |
| plot | ([DT:DT:MAXT], |     | error_u, | ’k:’);  |                                                         |     |     |     |     |     |     |     |     |
xlabel (’Time’);
ylabel (’Activity’);
| legend | (’\phi’, | ’\epsilon_p’, |     |     | ’\epsilon_u’); |     |     |     |     |     |     |     |     |
| ------ | -------- | ------------- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| axis   | ([0 MAXT | -2 3.5]);     |     |     |                |     |     |     |     |     |     |     |     |
Exercise4
Itiseasiesttoconsideravectoroftwonumbers(analogouscanbeshownforlongervectors):
 
x
| x=  | 1 . |     |     |     |     |     |     |     |     |     |     |     | (87) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
x
2
Theny=xTx=x2+x2,sothegradientisequalto:
1 2
∂ y
| ∂   |      |     |     |     |     |     |     |     |     |     |     |     |      |
| --- | ---- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
| y   | ∂ x  | 2 x   | .   |     |     |     |     |     |     |     |     |     |      |
| =  | ∂ 1 | = 1   | =2x |     |     |     |     |     |     |     |     |     | (88) |
| ∂ x |  y  |  2 x |     |     |     |     |     |     |     |     |     |     |      |
2
∂
|     | x 2 |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Exercise5
| function  | exercise5 |             |              |                      |            |                  |              |            |          |     |     |     |     |
| --------- | --------- | ----------- | ------------ | -------------------- | ---------- | ---------------- | ------------ | ---------- | -------- | --- | --- | --- | --- |
| mean_phi  | =         | 5; %        | mean of      | input                | from       | the current      | level        |            |          |     |     |     |     |
| Sigma_phi |           | = 2; %      | variance     | of                   | input from | the              | current      | level      |          |     |     |     |     |
| phi_above |           | = 5; %      | input        | from the             | level      | above            |              |            |          |     |     |     |     |
| DT =      | 0.01;     | %           | integration  |                      | step       |                  |              |            |          |     |     |     |     |
| MAXT      | = 20;     | %           | maximum      | time                 | considered |                  |              |            |          |     |     |     |     |
| TRIALS    | = 1000;   | %           | number       | of simulated         |            | trials           |              |            |          |     |     |     |     |
| LRATE     | = 0.01;   | %           | learning     | rate                 |            |                  |              |            |          |     |     |     |     |
| Sigma(1)  | =         | 1; %        | initializing |                      | the value  | of weight        |              |            |          |     |     |     |     |
| for       | trial     | = 2:TRIALS  |              |                      |            |                  |              |            |          |     |     |     |     |
|           | error(1)  | = 0;        |              | % initializing       |            | the prediction   |              | error      |          |     |     |     |     |
|           | e(1) =    | 0;          |              | % initializing       |            | the interneuron  |              |            |          |     |     |     |     |
|           | phi =     | mean_phi    | + sqrt(      | Sigma_phi)           |            | * randn;         |              |            |          |     |     |     |     |
|           | for i     | = 2:MAXT/DT |              |                      |            |                  |              |            |          |     |     |     |     |
|           | error(i)  | =           | error(i-1)   |                      | + DT *     | (phi - phi_above |              | - e(i-1)); |          |     |     |     |     |
|           | e(i)      | = e(i-1)    | +            | DT * (Sigma(trial-1) |            |                  | * error(i-1) | -          | e(i-1)); |     |     |     |     |
end
|     | Sigma(trial) | =   | Sigma(trial-1) |     | + LRATE | * (error(end)*e(end) |     |     | -   | 1); |     |     |     |
| --- | ------------ | --- | -------------- | --- | ------- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
end
| plot | (Sigma, | ’k’); |     |     |     |     |     |     |     |     |     |     |     |
| ---- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
xlabel (’Trial’);
ylabel (’\Sigma’);
References Friston,Karl(2005).Atheoryofcorticalresponses.PhilosophicalTransactionsofthe
RoyalSocietyB,360,815–836.
Bastos,AndreM.,Usrey,W.Martin,Adams,RickA.,Mangun,GeorgeR.,Fries,Pascal, Friston,Karl(2008).Hierarchicalmodelsinthebrain.PLoSComputationalBiology,
| &Friston,KarlJ.(2012).Canonicalmicrocircuitsforpredictivecoding.Neuron, |     |     |     |     |     |     |     | 4,e1000211. |     |     |     |     |     |
| ----------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
76,695–711. Friston,Karl(2010).Thefree-energyprinciple:aunifiedbraintheory?NatureRe-
Bell,AnthonyJ.,&Sejnowski,TerrenceJ.(1995).Aninformation-maximizationap- viewsNeuroscience,11,127–138.
proachtoblindseparationandblinddeconvolution.NeuralComputation,7, Friston,Karl,Schwartenbeck,Philipp,FitzGerald,Thomas,Moutoussis,Michael,
1129–1159. Behrens,Timothy,&Dolan,RaymondJ.(2013).Theanatomyofchoice:active
Bell,AnthonyJ.,&Sejnowski,TerrenceJ.(1997).Theindependentcomponentsof
inferenceandagency.FrontiersinHumanNeuroscience,7,598.
naturalscenesareedgefilters.VisionResearch,37,3327–3338. Harwood,David,Ojala,Timo,Pietikäinen,Matti,Kelman,Shalom,&Davis,Larry
Bogacz,Rafal,Brown,MalcolmW.,&Giraud-Carrier,Christophe(2001).Emergence
(1995).Textureclassificationbycenter-symmetricauto-correlation,usingKull-
ofmovementsensitiveneurons’propertiesbylearningasparsecodefornatural
movingimages.AdvancesinNeuralInformationProcessingSystems,13,838–844. backdiscriminationofdistributions.PatternRecognitionLetters,16,1–10.
|     |     |     |     |     |     |     |     | Olshausen, | Bruno | A., & Field, David | J. (1995). Emergence | of simple-cell | recep- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------------------ | -------------------- | -------------- | ------ |
Bogacz,Rafal,&Gurney,Kevin(2007).Thebasalgangliaandcorteximplement
optimaldecisionmakingbetweenalternativeactions.NeuralComputation,19, tivefieldpropertiesbylearningasparsecodefornaturalimages.Nature,381,
607–609.
442–477.
Chen,J.-Y.,Lonjers,P.,Lee,C.,Chistiakova,M.,Volgushev,M.,&Bazhenov,M.(2013). O’Reilly,RandallC.,&Munakata,Yuko(2000).Computationalexplorationsincogni-
tiveneuroscience.MITPress.
Heterosynapticplasticitypreventsrunawaysynapticdynamics.JournalofNeu-
roscience,33,15915–15929. Ostwald,Dirk,Kirilina,Evgeniya,Starke,Ludger,&Blankenburg,Felix(2014).Atu-
Feldman,Harriet,&Friston,Karl(2010).Attention,uncertainty,andfree-energy.
torialonvariationalBayesforlatentlinearstochastictime-seriesmodels.Jour-
FrontiersinHumanNeuroscience,4,215. nalofMathematicalPsychology,60,1–19.
FitzGerald,ThomasH.B.,Schwartenbeck,Philipp,Moutoussis,Michael,Dolan,Ray-
Rao,RajeshP.N.,&Ballard,DanaH.(1999).Predictivecodinginthevisualcortex:a
mondJ.,&Friston,Karl(2015).Activeinference,evidenceaccumulationandthe functionalinterpretationofsomeextra-classicalreceptive-fieldeffects.Nature
| urntask.NeuralComputation,27,306–328. |              |          |               |     |            |                  |     | Neuroscience,2,79–87. |     |     |     |     |     |
| ------------------------------------- | ------------ | -------- | ------------- | --- | ---------- | ---------------- | --- | --------------------- | --- | --- | --- | --- | --- |
| Friston,                              | Karl (2003). | Learning | and inference | in  | the brain. | Neural Networks, | 16, |                       |     |     |     |     |     |
Strogatz,Steven(1994).Nonlineardynamicsandchaos.WestviewPress.
1325–1352.