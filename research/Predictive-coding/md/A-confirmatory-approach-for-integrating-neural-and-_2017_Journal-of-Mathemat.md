JournalofMathematicalPsychology76(2017)131–141
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
A confirmatory approach for integrating neural and behavioral data
into a single model✩
∗
DonvanRavenzwaaij ,AlexanderProvost,ScottD.Brown
UniversityofNewcastle,Australia
h i g h l i g h t s
• Anoverviewofjointmodelingofbehavioralandneuraldata.
• AjointmodelingaccountofmentalrotationbehavioraldataandERPdata.
• Aformalcomparisonofseveraljointmodelingalternatives.
• Driftrateiscapableofsimultaneouslyexplainingbehavioraldataandneuraldata.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Recent decades have witnessed amazing advances in both mathematical models of cognition and in
Availableonline19May2016 thefieldofcognitiveneuroscience.Thesedevelopmentswereinitiallyindependentofoneanother,but
recentlythefieldshavestartedtobecomeinterestedinjoiningforces.Theresultingjointmodelingof
Keywords:
behavioralandneuraldatacanbedifficult,buthasprovedfruitful.Webrieflyreviewdifferentapproaches
Jointmodeling
usedindecision-makingresearchforlinkingbehavioralandneuraldata,andalsoprovideanexample.
Cognitiveneuroscience
Our example provides a tight link between behavioral data and evoked scalp potentials measured
Responsetimedata
ERP duringmentalrotation.Theexamplemodelillustratesapowerfulhypothesis-drivenwayoflinkingsuch
datasets.Wedemonstratetheuseofsuchamodel,provideamodelcomparisonagainstinteresting
alternatives,anddiscusstheconclusionsthatfollowfromapplyingsuchajointmodel.
©2016ElsevierInc.Allrightsreserved.
Like many areas of scientific enquiry, cognitive psychology the magnetoencephalogram (MEG; e.g., Brenner, Williamson, &
began with verbally-specified theories and gradually progressed Kaufman, 1975), functional magnetic resonance imaging (fMRI;
toquantitativeaccountsovertime.Thisresultedinmathematical e.g., Belliveau et al., 1991), and single-unit recordings in non-
models to describe memory (e.g., Atkinson & Shiffrin, 1968; humanprimates(e.g.,Hanes&Schall,1996;Schall,2001;Shadlen
Raaijmakers & Shiffrin, 1981), categorization (e.g., Nosofsky, & Newsome, 1996). As progressively more precise measures of
1986; Nosofsky & Palmeri, 1997), speeded and unspeeded the inner workings of the brain became available, researchers
decision making (e.g., Ratcliff, 1978; Wagenmakers, 2009), and have become increasingly capable at understanding the neural
many other paradigms (for reviews, see Lee & Wagenmakers, determinantsofcognitiveprocesses.
2013; Lewandowsky & Farrell, 2010). A separate and at first Some research paradigms have well-specified and tractable
mostly unrelated development was the advent of cognitive mathematical models of cognition, and also well-developed
neuroscience. This field sought to map changes in the brain as methods for neural measurement, for example, simple decision-
they related to cognition, using neural measurements obtained making and reinforcement learning. Researchers interested in
through event-related potentials (ERPs; e.g., Hillyard, Hink, suchparadigmsstartedinvestigatingwaystolinktheneuraland
Schwent, & Picton, 1973; Sutton, Braren, Zubin, & John, 1965), behavioral data more carefully. The latest developments include
so-called joint models, in which data of one kind can inform
the model fit of the other kind and vice versa (e.g., Anderson
& Fincham, 2014; Purcell et al., 2010; Turner, Forstmann, Love,
✩
ThisresearchwassupportedbyfellowshipgrantsfromtheAustralianResearch
Palmeri, & van Maanen, 2017; Turner, Forstmann et al., 2013).
CounciltoDVR(DE140101181)andSDB(FT120100244).
∗ These accounts aim for the most explicit and careful links,
Correspondingauthorat:UniversityofNewcastle,DepartmentofPsychology,
UniversityDrive,AviationBuilding,roomAVG11,CallaghanNSW2308,Australia. by simultaneously modeling neural recordings and behavioral
E-mailaddress:don.vanravenzwaaij@newcastle.edu.au(D.vanRavenzwaaij). outputs, allowing both kinds of data to inform model selection
http://dx.doi.org/10.1016/j.jmp.2016.04.005
0022-2496/©2016ElsevierInc.Allrightsreserved.

132 D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141
andparameterestimation.Jointmodelingprovidesanimportant to predict both behavioral measurements and a second set of
theoretical contribution: it allows a researcher to examine neuralmeasurements.Thisprovidesasimultaneousdescriptionof
common denominators underlying both behavioral data and neural and behavioral data sets, as well as explicating the links
neuraldata. between them. A less detailed, but more general approach was
Inthispaper,weprovideanexampleofhowtojointlymodel developed by Turner, Forstmann et al. (2013) and extended by
behavioral and neural data from simple decision-making. As Turner et al. (2017) in this volume. In their method, neural and
an illustrative example, we apply a joint model of behavioral behavioral models are joined by allowing their parameters to
responsesandEEGrecordingstodatafromanexperimentbased covary.Turner,Forstmann,etal.’sapproachisa‘‘joint’’model,in
on the classic Shepard–Metzler mental rotation task (Shepard & the sense that it allows symmetric information flow: behavioral
Metzler,1971).However,beforedescribingthemodel,wereview data can influence the neural parameter estimates, and neural
differentapproachestolinkingbehavioralandneuraldata,witha data can influence the behavioral parameter estimates. This
focusondecision-makingresearch. informationflowisachievedviaacovariancematrixforthemodel
An important change in the development of decision-making parameters.Thisstructureallowstheidentificationofcovariance
modelsoverthepasttwentyyearshasbeenasteady‘‘tightening’’ betweenmodelparametersassociatedwithneuralprocessesand
of the link between neural and behavioral data (for reviews modelparametersassociatedwithbehavioralprocesses.However,
and discussion of linking behavioral and neural data, see Teller, Turner,Forstmann,etal.’sapproachdiffersfromouranalysesinits
1984).Earlymodelsofsimpledecision-makinglinkedbehavioral focus.ThecovariancematrixofTurner,Forstmann,etal.’sapproach
and neural data loosely, by constraining the development of means that any and all parameters of the behavioral model are
behavioral models to respect data from neural measurements. allowedtolinkwithanyandallparametersoftheneuralmodel,
Forexample,theleakycompetingaccumulatormodeldeveloped although all these links are required to be linear. Our approach
by Usher and McClelland (2001) was structurally constrained to islessgeneral,butmorepointed,becauseitrequiresthespecific
include components supported by neural investigations, such as instantiationofasingle,preciselinkbetweenoneparameterofthe
lateralinhibitionbetweenaccumulatingunits,andpassivedecay neuralmodelandoneparameterofthebehavioralmodel.1
ofaccumulatedevidence.Theselinkswereincludedaspartofthe ThejointmodelingapproachofTurner,Forstmannetal.(2013)
modeldevelopmentprocess,andthereaftertherewasnofurther iscomplementarytotheapproachweuse.Forparadigmsinwhich
attempttolinkneuralwithbehavioraldata. there exist precise hypotheses about the links between neural
Subsequentmodelstestedthelinksviaqualitativecomparisons andbehavioralmodels,ourapproachoffersastraightforwardway
betweenpredictionsforcorrespondingneuralandbehavioraldata of instantiating and testing these hypotheses. For paradigms in
sets.Thiskindoflinkingwasverycommoninearlyresearchinto which this is not the case, Turner, Forstmann, et al.’s approach
decision-making with fMRI methods, in which predictions were offers a powerful tool for exploration. What both approaches
basedontheassumptionthatanexperimentalmanipulationwill haveincommonisthattheyjointlyfittheneuralandbehavioral
influenceoneparticularmodelcomponent,whichleadsnaturally data,whichallowsbehavioraldatatoinfluenceparametersonthe
to predictions for the behavioral data, and also for the neural ‘‘neural side’’ of the model, and vice versa. A joint model in this
data(viathehypothesizedlink).Predictionsmostfrequentlytake senseisabletoidentifyacompromisebetweenthetwostreamsof
the form ‘‘in condition A vs. B, behavioral measure X should data.Thismeansthat,comparedtoanotherwise-identicalmodel
increase while neural measure Y decreases’’. Support for the thatisfitsolelytothebehavioral(orneural)data,ajointmodel
predictionsistakenasevidenceinfavorofthemodel,including willalwaysfitmorepoorly.Coherentlymanagingthecompromise
the hypothesized link. As an example, Ho, Brown, and Serences betweenfittingneuralandbehavioraldatastreamsisastrength
(2009)testedpredictionsgeneratedfromdecision-makingmodels of the joint modeling approach. For example, suppose one was
via hypothesized neural links. In one part of their study, Ho examiningajointmodelforbehavioralandneuraldata,butwas
et al. manipulated the difficulty of a decision-making task and not fitting the model in a ‘‘joint’’ manner. Instead, imagine the
hypothesized that this should result in a change in the speed model was examined by fitting first to behavioral data alone,
of evidence accumulation in a sequential sampling model. By andthenlaterevaluatingthemodelbycomparingitssubsequent
examination of the model coupled to a standard model for predictionsforneuraleffectsagainsttheneuraldata.Oneproblem
hemodynamic responses, Ho et al. generated predictions for the withthisapproacharisesifthemodelhadtwosetsofparameters
blood–oxygen-level dependent (BOLD) response profile within (say,AandB)whichbothprovidedverygoodfitstothebehavioral
regions that are involved in perceptual decision making. These data, but very different fits to the neural data. Suppose that
predictionswerecomparedwithdatafromanfMRIexperiment, parametersetAprovidedslightlybetterbehavioralfits,butalso
whichlentsupporttosomeaccountsoverothers. terrible neural fits, while parameter set B provided good fits to
Linkingviathetestingofqualitativehypotheseswaslatersur- the neural data. Fitting to the behavioral data alone would lead
passedbyquantitativeapproaches,whichprovidedatighterlink theresearchertochooseparametersetA,andthentorejectthe
betweenneuralandbehavioraldata.Themostcommonexample modelbecauseoftheterriblefittoneuraldata.Jointfittingallows
of quantitative linking in decision-making models takes param- identification of compromise parameters (such as set B) which
eters of the decision-making model, estimated from behavioral providegoodfitstobothdatastreams.
data,andcomparesthemagainsttheparametersofadescriptive Thetwo-stageapproachtomodelevaluation,inwhichtheflow
model estimated from the neural data. For example, Forstmann ofinformationbetweenthetwotypesofdataismostlyone-way,
etal.(2008)correlatedindividualsubjects’modelparameters,es- was employed by Purcell et al. (2010) (they used two different
timatedfrombehavioraldata,againstblood–oxygen-leveldepen- neuraldatastreams,onlyoneofwhichwasafittingtarget).While
dent(BOLD)parameterestimates;subjectswithlargechangesin wehopethatajointmodelingapproachhassomestrengthsthat
thresholdparametersalsoshowedsimilarlylargechangesinBOLD the two-stage approach does not, Purcell et al.’s work included
responses.
Most recently, there have been efforts to link neural and
behavioraldecision-makingdataevenmoretightly,bycombining
1 While it is true that Turner, Forstmann, et al.’s method could, in theory,
both data sets in a single model-based analysis. This approach
berestrictedtoproduceourapproach(e.g.bysettingalmostallpriorsonthe
hasculminatedinmodelssuchasthatdevelopedbyPurcelletal.
covariancematrixcomponentstozero,andbyaddinginnonlinearparameterlink
(2010)whichusesneuralmeasurementsasamodelinputinorder functions)inpracticethishasnotbeendone.

D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141 133
Fig.1. TwosamplestimulifromProvostetal.(2013).Leftpanel:therightstimulusisthesameastheleftstimulus.Rightpanel:therightstimulusisdifferentfromtheleft
stimulus.
important other advantages that have been absent in the joint
modeling work to date. For example, Purcell et al.’s approach
was used to conduct pointed comparisons between competing
hypothesesaboutboththeunderlyingmodelstructures,andthe
hypothesized links between neural and behavioral data. While
such comparisons are, theoretically, possible in joint modeling
approaches,theycanbedifficulttoimplement,andhavenotbeen
investigated to date. The joint model we describe below is an
attempttocombinetheadvantagesoftheconfirmatoryapproach
ofPurcelletal.(2010)withthesophisticatedestimationapproach Fig.2. TheLBAanditsparametersfortworesponseoptions(forthistrial,the
ofTurner,Forstmannetal.(2013).SimilartoTurner,Forstmann, ‘‘different’’responseisthecorrectanswer). Evidenceaccumulationbeginsata
etal.’sapproach,weemployasimultaneousestimationprocedure. start point drawn randomly from a uniform distribution with interval [0, A].
Evidenceaccumulationisgovernedbydriftrated,drawnacrosstrialsfromanormal
However,ourapproachisconfirmatoryinthatwetestanexplicit distributionwithmeanνandstandarddeviations.Aresponseisgivenassoonas
andpre-specifiedlinkbetweenneuralandbehavioraldata.Wefit oneaccumulatorreachesthresholdb.ObservedRTisanadditivecombinationofthe
bothbehavioralandneuraldatastreamsatthesametime.Inthe timeduringwhichevidenceisaccumulatedandnon-decisiontimet0.
nextsection,wewilldescribethebehavioraltaskaswellasthetwo
typesofdata. The use of sequential accumulator models for the analysis of
responsetimedataisnotnew(e.g.,Link&Heath,1975;Ratcliff,
1. Data 1978;Wagenmakers,2009).Forthesedata,weturntoarelatively
recentaccumulatormodel:theLinearBallisticAccumulator(LBA;
The data we use are from an experiment based on the Brown&Heathcote,2008).AnadvantageoftheLBAistractability,
classic Shepard–Metzler mental rotation task (Provost, Johnson, as it has an easily-computed closed-form expression for its
Karayanidis,Brown,&Heathcote,2013).Thementalrotationtask likelihood. As a result, it is relatively straightforward to expand
is a two-alternative forced choice task in which participants are themodeltoincludeaneuralcomponent.Inthenextsection,we
asked to examine a pair of stimuli, one of which is rotated willintroducethereadertothebehavioralandneuralcomponents
relativetotheother.Crucially,participantsareaskedtoindicateas ofthemodelanddemonstratehowwecombinethemintoajoint
quicklyandaccuratelyaspossiblewhetherthestimuliareidentical model.
(‘‘same’’)orwhetheroneisdifferentfromtheother(‘‘different’’).
For instance, in the left panel of Fig. 1, the right stimulus is the 2. Themodeling
sameastheleftstimulus.Ontheotherhand,intherightpanelof
Fig.1,therightstimulusisthemirror-imageoftheleftstimulus In the first sub-section below, we introduce the behavioral
(‘‘different’’). model. In the second sub-section, we introduce the neural
The data we use here is from the first session of the first modelingandthelinkbetweenthetwoelements.
experiment reported by Provost et al. (2013). The experiment
includedfiveconditionsthatdifferedintheangleofrotationofthe 2.1. TheBehavioralLevel:LBA
rightstimulus:0°,45°,90°,135°,and180°.Theleftstimuluswas
alwaysidenticaltotheonedisplayedinFig.1. IntheLBAformulti-alternativeRTtasks(Brown&Heathcote,
Within each condition, half of the stimuli were ‘‘same’’ and 2008),stimulusprocessingisconceptualizedastheaccumulation
half were ‘‘different’’. The corresponding behavioral data were ofinformationovertime.Aresponseisinitiatedwhentheaccu-
responsetimesandchoicesfromallconditions,forallparticipants. mulatedevidencereachesapredefinedthreshold.Anillustration
The neural data we will consider are mean amplitudes of single fortworesponseoptionsisgiveninFig.2.
trials of the ERP signal corresponding to each trial used in the TheLBAassumesthatthedecisionprocessstartsfromarandom
behavioral analysis. As in Provost et al. (2013), we report ERP point between 0 and A, after which information is accumulated
effectsfromthemidlineparietalelectrodesitePz,withacommon linearly for each response option. The rate of this evidence
average reference. Comparing mean amplitudes at Pz we are accumulation is determined by drift rates d and d , normally
1 2
ν ν
able to model a specific ERP modulation called ‘‘rotation related distributedovertrialswithmeans and ,andcommonstandard
1 2
negativity’’ (RRN; Heil, 2002; Riečanský & Jagla, 2008), which is deviations.Thedistributionofdriftratesistruncatedatzeroto
considered an index of mental rotation. Specifically, we look at preventnegativeaccumulationrates.Thresholdbdeterminesthe
increased mean amplitude negativity associated with increased speed-accuracytradeoff;loweringbleadstofasterRTsatthecost
angulardisplacementacross8epochs,from200to1000mspost ofahighererrorrate.
stimulus onset in 100 ms windows. For more details, please see Together,theseparametersgenerateadistributionofdecision
themethodssectionofExperiment1andFigure4ofProvostetal. times DT. The observed RT, however, also consists of stimulus-
(2013). nonspecificcomponentssuchasresponsepreparationandmotor

134 D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141
| execution, | which | together | make | up non-decision |     | time | t . The |     |     |     |     |     |     |     |     |
| ---------- | ----- | -------- | ---- | --------------- | --- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
0
| modelassumesthatt |     | simplyshiftsthedistributionofDT,such |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
| thatRT =DT | +t  | (Luce,1986).Hence,thethreekeycomponents |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | --- | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
oftheLBAare(1)thespeedofinformationprocessing,quantified
ν
| bymeandriftrate |      | ;(2)responsecaution,quantifiedbyboundary   |      |          |                  |     |       |     |     |     |     |     |     |     |     |
| --------------- | ---- | ------------------------------------------ | ---- | -------- | ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
|                 |      |                                            | −    | /        |                  |     |       |     |     |     |     |     |     |     |     |
| separation      | that | averages                                   | to b | A 2; and | (3) non-decision |     | time, |     |     |     |     |     |     |     |     |
| quantifiedbyt   |      | .TheLBAhasbeensuccessfullyappliedtoanumber |      |          |                  |     |       |     |     |     |     |     |     |     |     |
0
| of experimental |     | paradigms | including | random |     | dot motion | tasks, |     |     |     |     |     |     |     |     |
| --------------- | --- | --------- | --------- | ------ | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
brightnessdiscrimination,consumerchoice,andmanyothers(e.g.,
Ho,Brown,Abuyo,Ku,&Serences,2012;Rae,Heathcote,Donkin,
Averell,&Brown,2014;Trueblood,Brown,&Heathcote,2014).
WespecifiedthestandardbehavioralaspectsoftheLBAmodel
using24parametersperparticipantforthe20differentresponse
timedistributions.Theparametersincluded:oneupperrangeof
startingpointAparameter,twoparametersforthresholdb(one
|          |          |                   |     |             |     |            |     | Fig. 3. A symbolic |     | display of | the joint | model. All | LBA parameters | inform | the |
| -------- | -------- | ----------------- | --- | ----------- | --- | ---------- | --- | ------------------ | --- | ---------- | --------- | ---------- | -------------- | ------ | --- |
| each for | ‘‘same’’ | and ‘‘different’’ |     | responses), | ten | parameters | for |                    |     |            |           |            |                |        |     |
v v behavioralresponsedata(bottom-left).Thedriftratecorrespondingtotheresponse
| both correct                                | drift         | c and              | for error | drift | e (two     | stimuli    | types—    |                                                       |              |            |     |          |         |            |     |
| ------------------------------------------- | ------------- | ------------------ | --------- | ----- | ---------- | ---------- | --------- | ----------------------------------------------------- | ------------ | ---------- | --- | -------- | ------- | ---------- | --- |
|                                             |               |                    |           |       |            | 0°,        | 45°, 90°, | givenbytheparticipantinformstheERPdata(bottom-right). |              |            |     |          |         |            |     |
| ‘‘same’’,                                   | ‘‘different’’ | – times            | five      | angle | conditions | –          |           |                                                       |              |            |     |          |         |            |     |
| 135°,180°),andonenon-decisiontimeparametert |               |                    |           |       |            | .The20dif- |           |                                                       |              |            |     |          |         |            |     |
|                                             |               |                    |           |       |            | 0          |           | Forsampling,weused32interactingMarkovchains,andran    |              |            |     |          |         |            |     |
| ferent response                             |               | time distributions |           | (and  | 10 free    | response   | proba-    |                                                       |              |            |     |          |         |            |     |
|                                             |               |                    |           |       |            |            |           | each for                                              | 1000 burn-in | iterations |     | followed | by 1000 | iterations |     |
bilities)arosefromfactorialcombinationoftwostimulusclasses
afterconvergence.Thetwotuningparametersofthedifferential
| (same vs. | different) | with | two response |     | classes | (same | vs. differ- |     |     |     |     |     |     |     |     |
| --------- | ---------- | ---- | ------------ | --- | ------- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
evolutionproposalalgorithmweresettostandardvaluesusedin
ent)andfiverotationangles.Thisparametrizationwaschosenbe-
previouswork:randomperturbationswereaddedtoallproposals
causeitprovidesareasonablecompromisebetweengoodness-of-
drawnuniformlyfromtheinterval[−0.001,0.001];andthescaleof
fitandtractability,asdemonstratedintheextensiveanalysesof thedifferenceaddedforproposalgenerationwassetto γ =2 . 38×
| alternativemodelsfordatafromarelatedexperiment(Provost& |     |     |     |     |     |     |     | ( )−0.5, |       |          |        |               |     |                 |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | ----- | -------- | ------ | ------------- | --- | --------------- | --- |
|                                                         |     |     |     |     |     |     |     | 2K       | where | K is the | number | of parameters |     | per participant |     |
Heathcote,2015).Importantly,theevidenceaccumulatorsofthe
|     |     |     |     |     |     |     |     | (24, in the | model | described | above). | The | MCMC | chains | blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ----- | --------- | ------- | --- | ---- | ------ | ------- |
modelhavebeenlinkedtoneuralactivityinthebrain(e.g.,Gold&
|     |     |     |     |     |     |     |     | proposals | separately | for each | participant’s |     | parameters, | and | also |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ---------- | -------- | ------------- | --- | ----------- | --- | ---- |
S hadlen,2007;Purcelletal.,2010).Becauseofthis,meandriftrate blockedthegroup-levelparametersin{µ,σ}pairs.
ν
lendsitselfnaturallytobethedrivingparameterbehindourERP
data.
We used a hierarchical Bayesian implementation of the LBA 2.2. Linkingtoneuraldata
| (Turner, | Sederberg, | Brown, | &   | Steyvers, | 2013). | Advantages | of  |     |     |     |     |     |     |     |     |
| -------- | ---------- | ------ | --- | --------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the hierarchical Bayesian framework include the ability to fit The behavioral model above, based on the LBA, specifies a
the LBA to data with relatively few trials, because the model likelihood function for the response time data which gives the
|               |          |           |                  |           |            |       |           | likelihood | of observed | data            | conditional |          | on any   | given | set of |
| ------------- | -------- | --------- | ---------------- | --------- | ---------- | ----- | --------- | ---------- | ----------- | --------------- | ----------- | -------- | -------- | ----- | ------ |
| borrows       | strength | from      | the hierarchical |           | structure. | The   | Bayesian  |            |             |                 |             |          |          |       |        |
|               |          |           |                  |           |            |       |           | parameter  | values.     | This likelihood |             | function | supports | all   | of our |
| set-up allows |          | for using | MCMC             | sampling, | which      | is an | efficient |            |             |                 |             |          |          |       |        |
statisticalanalyses.Thefirststepinbringingtheneuraldatainto
| approach | to parameter |     | estimation | (Gamerman |     | & Lopes, | 2006; |           |       |          |            |          |     |           |     |
| -------- | ------------ | --- | ---------- | --------- | --- | -------- | ----- | --------- | ----- | -------- | ---------- | -------- | --- | --------- | --- |
|          |              |     |            |           |     |          |       | the model | is to | define a | likelihood | function | for | the ERPs. | We  |
Gilks,Richardson,&Spiegelhalter,1996;vanRavenzwaaij,Cassey,
willassumethattheERPdata,withinanyparticularconditionfor
| & Brown, | in press). | Starting | points | for | the Markov | chains  | were |     |     |     |     |     |     |     |     |
| -------- | ---------- | -------- | ------ | --- | ---------- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
|          |            |          |        |     | (          | , . )|( | ,)   |     |     |     |     |     |     |     |     |
drawnfromthefollowingdistributions:A ∼ N 2 0 2 0 ,both anyparticularsubject,arenormallydistributed.Thenextstepis
bs ∼ N ( 1 , 0 . 1 )|( 0 ,) ,allten ν s ∼ N ( 3 , 0 . 3 )|( 0 ,) ,allten ν s ∼ to link the parameters of the behavioral LBA model above with
|     |     |     | c   |     |     |     | e   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
N ( 1 , 0 . 1 )|( 0 ,) , and t ∼ N ( 0 . 2 , 0 . 02 )|( 0 ,) . In this notation, ∼ the parameters of the assumed normal model for the ERP data.
0
N ( x , y )|( 0 ,) indicates that a parameter is normally distributed To begin, we assume that the standard deviation of the normal
with mean x, standard deviation y, and is truncated to positive distribution is fixed everywhere, for each subject, and that the
valuesonly. mean of the normal distribution is given by an offset parameter
|                  |     |        |            |      |                |     |        | α                                                 |     |     |     |     |     | β   |     |
| ---------------- | --- | ------ | ---------- | ---- | -------------- | --- | ------ | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                  |     |        |            |      |                |     |        | ( )plusthedriftrateparametertimesascaleparameter( |     |     |     |     |     |     | ):  |
| The hierarchical |     | set-up | prescribes | that | all individual |     | param- |                                                   |     |     |     |     |     |     |     |
eters come from a truncated Gaussian group-level distribution. ∼N (α+v×β,σ).
|     |     |     |     |     |     |     |     | ERP |     |     |     |     |     |     | (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thus,foreachparametertobeestimated,weestimatedagroup
levelmeanparameterandagrouplevelstandarddeviationparam- The model is graphically displayed in Fig. 3. Eq. (1) provides a
eter.Priorsforallgrouplevelmeanparameterswerenormaldis- preciseinstantiationofthelinkinghypothesisinthisjointmodel.
|     |     |     | ( , )|( | ,)  |     | ( , | )|( ,) |     |     |     |     |     |     |     |     |
| --- | --- | --- | ------- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
tributions,withAµ ∼ N 2 1 0 ,bothbµ ∼ N 2 1 0 ,all Ourverysimplehypothesisisthattheneuralandbehavioraldata
ten ν cµ ∼ N ( 3 , 1 )|( 0 ,) , all ten ν eµ ∼ N ( 1 , 1 )|( 0 ,) , and t 0µ ∼ are linked via the drift rate parameter of the model, and that
N ( 0 . 2 , 0 . 1 )|( 0 ,) .Priorsforallgrouplevelstandarddeviationpa- thelinkisasimplelinearfunction.Whilesimpletospecify,this
rametersweregammadistributionswithashapeandascalepa- linkhascomplicatedimplications.Forexample,thepredictedERP
rameterof1,exceptfort whichhasascaleparameterof3.Start- signalwillchangeacrossconditionswheneverdriftratechanges—
0σ
|                                    |     |     |     | µ   |                          |     |     | withrotationangleandwithsamevs.differentstimuluspairings, |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ------------------------ | --- | --- | --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| ingpointdistributionsforgrouplevel |     |     |     |     | wereallidenticaltostart- |     |     |                                                           |     |     |     |     |     |     |     |
ingpointdistributionsfortheindividualparameters,andstarting in our experiment. The link also implies particular constraints
σ
pointdistributionsforgrouplevel parameterswerederivedfrom on the model. For example, the drift rate parameter is forced to
startingpointdistributionsfortheindividualparametersbydivid- accommodatechangesinbothbehavioralandneuraldatadueto
ingthemeanby10andthestandarddeviationby2.Theseprior changesinrotationangle.Thelinkingparameterservesasatime-
settingsarequiteuninformative,andarebasedonpreviousexpe- sensitivemeasureofthelinkbetweenbehavioralandneuraldata.
riencewithparameterestimationfortheLBAmodel.Asaresult, The model does more than just re-describe this link: the model
thespecificsettingswillnothavealargeinfluenceontheshape attempts to capture the fact that different rotation angles cause
oftheposterior.Formoredetailsondistributionalchoicesforthe different ERP measurements with a linking function and linking
priors,wereferthereadertoTurner,Sederbergetal.(2013). parameters that are identical for all angle conditions. As such,

D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141 135
Posteriorpredictivesshowthattheν-ERPmodelfitboththebehavioraldataandtheneuraldatawell.Lefttwocolumns:proportioncorrect(y-axis)plottedagainst
Fig.4.
RTs(x-axis)forthe0.1,0.3,0.5,0.7,and0.9quantilescalculatedfromcorrectRTs(green)anderrorRTs(red).Righttwocolumns:meanERPamplitudesinnegativemicrovolts
(y-axis)foreightdifferenttimewindows(x-axis).Forallpanels,boxplotsrepresentempiricaldataandlinesrepresentposteriorpredictivedata.(Forinterpretationofthe
referencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
µ
the model accounts for different ERPs across conditions entirely pointdistributionsforgrouplevel wereallidenticaltostarting
throughdriftrate. point distributions for the individual parameters, and starting
|     |     |     |     |     |     |     | α   |     |     | σ   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Fortheneuraldata,weestimatedoneoffsetparameter ,one point distributions for group level parameters were derived
|     |     |     |     | σ   |     |     | β   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
standard deviation parameter , and eight scale parameters fromstartingpointdistributionsfortheindividualparametersby
(oneforeach100msepochfrom200upto1000ms).Theeight dividingthemeanby10andthestandarddeviationby2.
scaleparametersallowinvestigationofhowstronglytheERPsignal Dataandcodeforthefullmodelmaybefoundontheweb.2
| is linked | to cognition |     | across | the eight | different | time | windows |     |     |     |     |     |
| --------- | ------------ | --- | ------ | --------- | --------- | ---- | ------- | --- | --- | --- | --- | --- |
...
| (200–300ms,300–400ms, |     |     |     | ,900–1000ms). |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
2.3. Modelcomparison
| Starting      | points         | for | the linking | parameters |           | were drawn  | from |     |     |     |     |     |
| ------------- | -------------- | --- | ----------- | ---------- | --------- | ----------- | ---- | --- | --- | --- | --- | --- |
|               |                |     | α           |            | ( , . )|( | ,)          | β    |     |     |     |     |     |
| the following | distributions: |     |             | ∼ N        | 8 0 8 0   | , all eight | s ∼  |     |     |     |     |     |
Itisverydifficulttojudgemodelfitinanabsolutesense.What
| N ( 1 , 0 . 1 )|( | 0 ,) , | and σ | ∼ N ( | 5 , 0 . 5 )|( | 0 ,) . Analogous |     | to the LBA |             |                 |                  |                |     |
| ----------------- | ------ | ----- | ----- | ------------- | ---------------- | --- | ---------- | ----------- | --------------- | ---------------- | -------------- | --- |
|                   |        |       |       |               |                  |     |            | constitutes | a good fit, how | much of a misfit | is acceptable? | In  |
parameters,allindividuallinkingparametersweredrawnfroma
practice,itisalmostalwaysmorefruitfultoexaminecomparative
| truncated  | Gaussian   | group-level |            | distribution. | Priors         | for  | all group |                  |        |                   |                    |     |
| ---------- | ---------- | ----------- | ---------- | ------------- | -------------- | ---- | --------- | ---------------- | ------ | ----------------- | ------------------ | --- |
|            |            |             |            |               |                |      |           | goodness-of-fit, | and to | compare different | models. We compare |     |
| level mean | parameters |             | are normal |               | distributions, | with | α ∼       |                  |        |                   |                    |     |
µ
| N ( 8 , 2 )|( | 0 ,) ,alleight | β   | ∼ N | ( 1 , 1 )|( | 0 ,) ,and σ | ∼ N ( | 5 , 1 )|( 0 ,) | .   |     |     |     |     |
| ------------- | -------------- | --- | --- | ----------- | ----------- | ----- | -------------- | --- | --- | --- | --- | --- |
|               |                |     | µ   |             |             | µ     |                |     |     |     |     |     |
Priorsforallgrouplevelstandarddeviationparametersaregamma
| distributions | with | a shape | and | a scale | parameter | of  | 1. Starting | 2   |     |     |     |     |
| ------------- | ---- | ------- | --- | ------- | --------- | --- | ----------- | --- | --- | --- | --- | --- |
www.donvanravenzwaaij.com/Papers,andalsoathttps://osf.io/2r7bv/.

136 D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141
|     |     |     |     | • ν -nonlinear-ERP: |            | identical |           | to the | ν -ERP        | model, | but testing | a       |
| --- | --- | --- | --- | ------------------- | ---------- | --------- | --------- | ------ | ------------- | ------ | ----------- | ------- |
|     |     |     |     | nonlinear           | link       | function  | between   |        | the drift     | rates  | and         | the ERP |
|     |     |     |     | mean                | parameter. | The       | nonlinear |        | link function |        | we test     | is the  |
cumulativenormaldistributionfunction,whichinstantiatesthe
hypothesisthatscalppotentialsmighthaveimportantceiling
andflooreffects.Sucheffectsareplausibleformanyreasons,
forexampletheymaybeimposedbyphysicalandphysiological
limitsontheelectricalactivityandconductivityofthecortex
andscalp.
Priorsandstartingvalueswereanalogousforallfourmodels.3
Themodelswillbecomparedbyvisuallyinspectingtheposterior
|     |     |     |     | predictives | for            | obvious | misfit.      | Numerically, |             | we  | compare   | the   |
| --- | --- | --- | --- | ----------- | -------------- | ------- | ------------ | ------------ | ----------- | --- | --------- | ----- |
|     |     |     |     | models      | by calculating |         | the Deviance |              | Information |     | Criterion | (DIC; |
Spiegelhalter,Best,Carlin,&vanderLinde,2002),ameasurewhich
balancesgoodnessoffitagainstmodelcomplexity.Inthissense,
|     |     |     |     | DIC is similar | to  | the well-known |     | BIC | and | AIC measures, |     | but DIC |
| --- | --- | --- | --- | -------------- | --- | -------------- | --- | --- | --- | ------------- | --- | ------- |
extendsthesebyquantifyingmodelcomplexityasacross-sample
variabilityinmodelfitratherthansimplycountingupthenumber
offreeparameters.Assuch,DICusuallyassumesastrongerpenalty
forcomplexity.LowervaluesofDICindicatebettersupportfora
modelfromthedata.
3. Results
3.1. ν -ERP
Asafirstcheckofmodelfit,wecomparedposteriorpredictive
|     |     |     |     | data against         | the       | neural        | and        | behavioral |                    | data           | in Fig.       | 4. The  |
| --- | --- | --- | --- | -------------------- | --------- | ------------- | ---------- | ---------- | ------------------ | -------------- | ------------- | ------- |
|     |     |     |     | figure displays      |           | data averaged |            | over       | participants       |                | with boxplots |         |
|     |     |     |     | representing         | empirical |               | data       | and        | lines representing |                | synthetic     |         |
|     |     |     |     | data. The            | left      | two columns   |            | show       | model              | correspondence |               | to      |
|     |     |     |     | the 0.1,             | 0.3, 0.5, | 0.7,          | and 0.9    | quantiles  | calculated         |                | from          | correct |
|     |     |     |     | RTs (green)          | and       | error         | RTs (red). | The        | right              | two            | columns       | show    |
|     |     |     |     | model correspondence |           |               | to mean    | ERP        | amplitudes         |                | for           | each of |
|     |     |     |     | the eight            | different | time          | windows    |            | (200–300           | ms,            | 300–400       | ms,     |
|     |     |     |     | ... , 900–1000       |           | ms).          | The first  | and        | third              | column         | show          | model   |
correspondencefor‘‘same’’stimuli,thesecondandfourthcolumn
|     |     |     |     | show model           | correspondence |     |               | for ‘‘different’’ |          | stimuli.   | Rows | show      |
| --- | --- | --- | --- | -------------------- | -------------- | --- | ------------- | ----------------- | -------- | ---------- | ---- | --------- |
|     |     |     |     | model correspondence |                |     | for different |                   | rotation | conditions |      | (0°, 45°, |
90°,135°,180°).
|     |     |     |     | Onthewhole,the |     | ν   | -ERPmodelfitbothdatasetswell,although |     |     |     |     |     |
| --- | --- | --- | --- | -------------- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- |
thereissomemisfit.Themodelcapturesthequalitativechangesin
RTdistributionsandpercentagecorrectacrosssamevs.different
|     |     |     |     | stimuli,    | and across | the       | different        | angles   | of   | rotation.    | There | is a    |
| --- | --- | --- | --- | ----------- | ---------- | --------- | ---------------- | -------- | ---- | ------------ | ----- | ------- |
|     |     |     |     | tendency    | for the    | model     | to under-predict |          |      | the accuracy |       | in some |
|     |     |     |     | conditions, | as         | evidenced | by               | the fact | that | the green    | lines | are     |
slightlylowerthanthecenterofthegreenboxplotsandthered
Posteriorpredictivesshowthatthebehavioral-onlyversionofthejointν-
Fig.5. lines slightly higher than the center of the red boxplots. For the
ERPmodelfitthebehavioraldatawell.Displayedaretheproportioncorrect(y-axis) neuraldata,themodelseemstocapturetheERPdistributionsover
plottedagainstRTs(x-axis)forthe0.1,0.3,0.5,0.7,and0.9quantilescalculated
timewell.Theseconclusionsaboutabsolute,globalmodelfitare
fromcorrectRTs(green)anderrorRTs(red).Forbothpanels,boxplotsrepresent
necessarilyvague,becauseofthepreviouslymentioneddifficulties
empiricaldataandlinesrepresentposteriorpredictivedata.(Forinterpretationof
thereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversion inassessingabsolutemodelfit.Thisisoneofthereasonsweturn
| ofthisarticle.) |     |     |     | tomodelcomparisonbelow. |       |            |     |     |         |       |               |     |
| --------------- | --- | --- | --- | ----------------------- | ----- | ---------- | --- | --- | ------- | ----- | ------------- | --- |
|                 |     |     |     | The first               | model | comparison |     | we  | provide | is to | a behavioral- |     |
themodeldescribedabove(henceforth ν -ERP)tothreecompeting data-only version of the ν -ERP model. The joint model must
alternatives: necessarily fit more poorly than the behavioral-only model,
becausetheparametersofthejointmodelarefurtherconstrained
| • t -ERP:thebehavioralparametrizationisidenticaltothatof |     |     | ν - |                                                          |     |     |     |     |     |     |     |     |
| -------------------------------------------------------- | --- | --- | --- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0                                                        |     |     |     | toaccommodateeffectsintheneuraldata(inastatisticalsense, |     |     |     |     |     |     |     |     |
ERP,butthelinkingparametertotheneuraldataisnon-decision
|                          |     |                                |     | the behavioral-only                                    |     | model | ‘‘nests’’ |     | the behavioral |     | side of | the ν - |
| ------------------------ | --- | ------------------------------ | --- | ------------------------------------------------------ | --- | ----- | --------- | --- | -------------- | --- | ------- | ------- |
| timet insteadofdriftrate |     | ν (seee.g.Pougetetal.,2011,for |     |                                                        |     |       |           |     |                |     |         |         |
| 0                        |     |                                |     | ERP model).Inordertoexaminethisconstraint,wecomparethe |     |       |           |     |                |     |         |         |
corroboratingevidence).
posteriorpredictivesofourjointmodelfitstoposteriorpredictives
•
Brev–t 0 -ERP:thebehavioralparametrizationfordriftratesand of a fit to the behavioral data alone. Parameter settings were
ν
non-decisiontimearereversed.Inthismodel,wehaveone c asoutlinedinsection‘‘TheBehavioralLevel:LBA’’.Theposterior
ν
and one e instead of ten each, and we have ten t 0 (one for predictivesforthebehavioral-onlymodelcanbefoundinFig.5.
eachstimulusandanglecondition)insteadofone.Analogous
| to t 0 -ERP, | the linking parameter | to the neural | data is non- |     |     |     |     |     |     |     |     |     |
| ------------ | --------------------- | ------------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ν
| decisiontimet | .Analogousto | -ERP,thelinkingparameterto |     |     |     |     |     |     |     |     |     |     |
| ------------- | ------------ | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0 3 E . g ., a l l t e n t u s e d i n t h e B r e v – t - E R P m o d e l h a v e t h e s a m e s t a r t in g v al u e s a n d
t h e n e u r a l d a t a i snowfreetovarybetweenstimulitypesand 0 ν 0 ν ν
|                |                |     |     | p r io r a s t h | e o n e t 0 u | s e d i n t h   | e - E R P m   | o d e l ;a l | l te n c u  | s e d i n t h | e - E R P m     | o d e l h a v e |
| -------------- | -------------- | --- | --- | ---------------- | ------------- | --------------- | ------------- | ------------ | ----------- | ------------- | --------------- | --------------- |
| a n gl e c o n | d i ti o n s . |     |     |                  |               |                 |               |              | ν           |               |                 |                 |
|                |                |     |     | t h e s a m e s  | ta r ti n g v | a l u e s a n d | p r i o r a s | t h e o n e  | c u s e d i | n t h e B r e | v – t 0 - E R P | m o d e l.      |

D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141 137
Fig.6. Posteriorpredictivedatashowthatthet0-ERPmodelfitsthebehavioraldatawell,andtheneuraldatacomparativelypoorly.Lefttwocolumns:proportioncorrect(y-
axis)plottedagainstRTs(x-axis)forthe0.1,0.3,0.5,0.7,and0.9quantilescalculatedfromcorrectRTs(green)anderrorRTs(red).Righttwocolumns:meanERPamplitudes
innegativemicrovolts(y-axis)foreightdifferenttimewindows(x-axis).Forallpanels,boxplotsrepresentempiricaldataandlinesrepresentposteriorpredictivedata.(For
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
Visual comparison of Figs. 4 and 5 shows that the model fit first has a behavioral parametrization which is identical to the
isalmostidentical.Thejointmodelcompromisesslightlyonthe original ν -ERP model, but has a link to the neural data through
accuracyfitcomparedtothebehavioral-onlymodel,butotherthan thet parameter(thismodeliscalledt -ERP).Thesecondmodel
0 0
that,themodelsappearindistinguishable.Toprovideastatistical correspondstotheoriginal ν -ERPmodelbutwiththerolesfornon-
comparison,thelikelihoodofthemeanparametersaveragedover decisiontime(t )anddriftrate( ν )reversed(thismodeliscalled
0
all participants is −233.93 for the simple model. For the joint Brev–t -ERP).
0
model, when selecting the behavioral component of the model
andtakingthelikelihoodofthosemeanparametersaveragedover
allparticipants,thevalueis−255.69,lendingfurthercredenceto 3.2. t 0 -ERP
the observation that these models fit the data comparably. It is
notappropriatetocomparetheselikelihoodvaluesfurther,e.g.by Posterior predictive data for the t 0 -ERP model are shown in
calculatingDIC,becausethelikelihoodofthebehavioraldataunder Fig.6.Visualinspectionofthefigureshowsthatthet 0 -ERPmodel
thejointmodeldoesnotsatisfytheassumptionsofthoseanalyses, fits the behavioral data well, but does not capture the neural
ν
becauseoftheconditioningonneuraldata. data as well as the -ERP model. This impression is supported
ν
Wenextexamineadifferent,butplausible,candidateforthe by comparison of the DIC values for the two models: -ERP
link between behavioral and neural data: the non-decision time hasanaverageDICacrossparticipantsof31,880.75,whereast -
0
parameter, t . We do so by comparing two new models. The ERP has an average DIC across participants of 31,909.18. Within
0

138 D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141
Fig.7. PosteriorpredictivedatashowthattheBrev–t0-ERPmodelfitsboththebehavioralandtheneuraldataquitepoorly.Lefttwocolumns:proportioncorrect(y-axis)
plottedagainstRTs(x-axis)forthe0.1,0.3,0.5,0.7,and0.9quantilescalculatedfromcorrectRTs(green)anderrorRTs(red).Righttwocolumns:meanERPamplitudesin
negativemicrovolts(y-axis)foreightdifferenttimewindows(x-axis).Forallpanels,boxplotsrepresentempiricaldataandlinesrepresentposteriorpredictivedata.(For
interpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
participants, the ν -ERP model was DIC-preferred for 6 out of 9 3.4. ν -nonlinear-ERP
people.
ν
|     |     |     |     | Posteriorpredictivedataforthe |     | -nonlinear-ERP | areshownin |
| --- | --- | --- | --- | ----------------------------- | --- | -------------- | ---------- |
ν
|     |     |     |     | Fig.8.Visualinspectionofthefigureshowsthatthe |     |     | -nonlinear- |
| --- | --- | --- | --- | --------------------------------------------- | --- | --- | ----------- |
3.3. Brev–t -ERP ERP model fits both types of data well, although not as well as
0
|     |     |     |     | the ν -ERP model:   | ν -ERP has       | an average DIC across | participants |
| --- | --- | --- | --- | ------------------- | ---------------- | --------------------- | ------------ |
|     |     |     |     | of31,880.75,whereas | ν -nonlinear-ERP | hasanaverageDICacross |              |
PosteriorpredictivedatafortheBrev–t -ERPmodelareshown participants of 31,910.22. Within participants, the ν -ERP model
0
inFig.7.Visualinspectionofthefigureshowsthatthemodelfits
wasDIC-preferredfor6outof9people.Fortheremainderofthe
ν
thebehavioraldataworsethanbothothermodels.TheBrev–t 0 - resultssection,wewillexaminetheresultsof -ERP,thebestofthe
ERPmodelcapturestheneuraldatabetterthanthet 0 -ERPmodel, modelswehaveinvestigated,inmoredetail.
ν
| but not as good                       | as the -ERP | model. Again, | this impression   | is                   |     |     |     |
| ------------------------------------- | ----------- | ------------- | ----------------- | -------------------- | --- | --- | --- |
| supportedbyanalysisofDICvalues:Brev–t |             |               | -ERP hasanaverage |                      |     |     |     |
|                                       |             |               | 0                 | 3.5. Centralfindings |     |     |     |
DICacrossparticipantsof32,039.64,worsethanbothothermodels.
ItisalsothemodelwiththepoorestDICoutofallmodelsforall Insum,wefindthatofthemodelsweconsidered,evidencefor
nineparticipants. meandriftrate ν beingthelinkingparameterbetweenbehavioral

D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141 139
Fig.8. Posteriorpredictivedatashowthattheν-nonlinear-ERPmodelfitsthebehavioraldatawell,andtheneuraldatacomparativelypoorly.Lefttwocolumns:proportion
correct(y-axis)plottedagainstRTs(x-axis)forthe0.1,0.3,0.5,0.7,and0.9quantilescalculatedfromcorrectRTs(green)anderrorRTs(red).Righttwocolumns:meanERP
amplitudesinnegativemicrovolts(y-axis)foreightdifferenttimewindows(x-axis).Forallpanels,boxplotsrepresentempiricaldataandlinesrepresentposteriorpredictive
data.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
and neural data is strongest. Furthermore, we find that the Thetop-rightpaneldisplaysERPsforeach100msepochfrom
ν
relationshipbetweenmeandriftrate andtheneuraldataislinear 200upto1000msfor‘‘same’’stimuli.TheamplitudeoftheERPs
innature(thoughforsomeparticipants,anonlinearlinkprovides dropsastherotationangleincreases.Giventhesizeoftheerror
abetteraccountofthedata). bars(displayedinthefarrightofthepanel),themismatchbetween
Tohighlightthecentralresearchfindings,wewillnowexamine thedataandthemodelismodest.Thebottom-rightpaneldisplays
effectsacrossconditions.SummarizeddataaredisplayedinFig.9, ERPsforeach100msepochfrom200upto1000msfor‘‘different’’
withcorrespondingsummariesfromtheposteriorpredictionsof stimuli. Again, the amplitude of the ERPs drops as the rotation
thebest-supportedjointmodel, ν -ERP.Thetop-leftpaneldisplays factorincreases,thoughinterestinglytheERPsfor135°and180°
medianRTsfordifferentconditionsandstimulustypes.RTsteadily havereversedorder.Themodelcapturesthedatawell,ascanbe
increases as the rotation angle increases, and also median RT is observedbycomparingthemodestmismatchbetweendataand
higherfor‘‘different’’stimulithanfor‘‘same’’stimuli.Themodel modeltotheaveragederrorbarsdisplayedinthefarrightofthe
capturesbothofthesedatapatternsveryaccurately.Thebottom- panel.
leftpaneldisplaysmeanproportionofcorrectdecisions,separately Thebottom-rightpanelalsoincludesmediansoftheposterior
β
fordifferentconditionsandstimulustypes.Accuracydropsasthe distributionoverthegrouplevellinkingparameter( ),witherror
rotation angle increases, though the trend is less clear than for bars capturing the central 50% of the distribution. The size of
β
RTs. The bottom-left panel confirms the earlier observation that linking parameter follows the amplitude of the rotation-angle
themodelunderestimatessomeoftheaccuracies. effects in the ERP data. To reiterate, estimates of the linking

140 D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141
Fig.9. Behavioralandneuraldata,displayedforeachstimulustypeandanglecondition.Top-leftpanel:BoxplotsdisplaymedianRTforparticipantsinseconds.Bottom-left
panel:Boxplotsdisplayproportioncorrectforparticipants.Rightpanels:ERPsinnegativemicrovolts.Thefarrightofbothpanelsdisplayserrorbars,averagedoveralltime
slots.Insetforthebottom-rightpanelshowslinkingparameterβdisplayedforeachtimeslotwitherrorbars.Errorbarsintherightpanelsrepresentthecentral50%ofthe
distribution.
parameter provide time-sensitive measures of the link between The joint modeling approach that we have used relies on
behavioralandneuraldata.Forexample,ateachtimewindowthe the precise instantiation of hypotheses about the links between
differentrotationanglesleadtodifferentERPmeasurements(the parameters related to neural data and parameters related to
coloreddotsverticallyspaced),withsometimewindowsshowing behavioral data. In addition to specifying which parameters are
verylittledifferencesbetweenanglesandsomeshowingverylarge linked, this approach also requires specification of a particular
differences. The model captures these effects, even though the linking function. The different model versions we investigated
linkingfunctionandlinkingparametersareidenticalforallangle differed in these elements, and provided a rigorous framework
conditions. This happens because the drift rates are estimated forinvestigatingimportanttheoreticalquestions.Forexample,our
differently for the different angle conditions (and for same vs. modelcomparisonsrevealedthat,formostparticipantsatleast,a
differentstimulusclasses),andthesedifferentdriftratesinfluence simplelinearlinkbetweendriftrateandaverageERPamplitude
theERPpredictionsviathelinkingfunction. was better than a sigmoidal link. Similarly, an explanation of
rotation angle effects in both behavioral and neural data was
4. Conclusion better when based on drift rates than on non-decision time. It
is, of course, entirely conceivable that both drift rate and non-
This paper provided an example for cognitive scientists who decision time play a role in linking behavioral data to ERP data
ν
areinterestedininvestigatingthecorrespondencebetweenneural (recallthatDICpreferredthe -ERP modeloverthet 0 -ERP model
and behavioral data via building computational models for both formost,butnotall,participants).ProvostandHeathcote(2015)
data streams. We compared four different models that differ exploredmoresophisticatedmodelsusingasequential,two-stage
in the parametrization on the behavioral level and in the process,toseparatethedecisionprocessfromthementalrotation
linking assumptions and showed that drift rate is capable of process. In Provost et al.’s account, the decision processing is
simultaneouslyexplainingthebehavioraldataandtheneuraldata. delayedbyarandomamountoftimetakentomentallyrotatethe

D.vanRavenzwaaijetal./JournalofMathematicalPsychology76(2017)131–141 141
stimulus,whichisequivalenttoassumingarandomdistribution Forstmann,B.,Dutilh,G.,Brown,S.,Neumann,J.,vonCramon,D.,Ridderinkhof,
forthenon-decisiontime(t )inourmodel.Provostetal.’sanalyses K.,etal.(2008).Striatumandpre-SMAfacilitatedecision-makingundertime
0
pressure.ProceedingsoftheNationalAcademyofSciences,105,17538–17542.
supportedmodelswithvariablenon-decisiontimeprocesses,and
Gamerman,D.,&Lopes,H.F.(2006).MarkovchainMonteCarlo:stochasticsimulation
inparticularthosewherethevariabilityincreasedwiththemean. forBayesianinference.BocaRaton,FL:Chapman&Hall/CRC.
Aninterestingavenueforfutureresearchwouldbetoextendthe Gilks,W.R.,Richardson,S.,&Spiegelhalter,D.J.(Eds.)(1996).MarkovchainMonte
variousmodelcomparisonswehavemade(above)tomodelswith Carloinpractice.BocaRaton,FL:Chapman&Hall/CRC.
Gold,J.I.,&Shadlen,M.N.(2007).Theneuralbasisofdecisionmaking.Annual
sophisticatedrandomdistributionsfornon-decisiontime. ReviewofNeuroscience,30,535–574.
Our approach to linking neural and behavioral data is not Hanes,D.P.,&Schall,J.D.(1996).Neuralcontrolofvoluntarymovementinitiation.
unique, and is not necessarily the best for many different Science,274,427–430.
Heil,M.(2002).ThefunctionalsignificanceofERPeffectsduringmentalrotation.
situations. Still, we propose that, when possible, researchers
Psychophysiology,43,401–412.
should strive for the tightest possible linking, as this provides Hillyard,S.A.,Hink,R.F.,Schwent,V.L.,&Picton,T.W.(1973).Electricalsignsof
the greatest opportunity to investigate the underlying linking selectiveattentioninthehumanbrain.Science,182,177–180.
Ho,T.C.,Brown,S.,Abuyo,N.A.,Ku,E.-H.J.,&Serences,J.T.(2012).Perceptual
assumptions.Putdifferently,jointmodelsthataretightlylinked
consequences of feature-based attentional enhancement and suppression.
allow us to uncover underlying psychological processes that JournalofVision,12,1–17.
simultaneously explain behavioral and neural data. Such an Ho, T. C., Brown, S., & Serences, J. T. (2009). Domain general mechanisms of
perceptualdecisionmakinginhumancortex.TheJournalofNeuroscience,29,
approach is arguably more powerful than more loosely linked
8675–8687.
models, which often do not go beyond correlations between Lee, M. D., & Wagenmakers, E.-J. (2013). Bayesian cognitive model-
differentlevelsofdata.The‘‘whenpossible’’caveatrelatestothe ing: a practical course. Cambridge University Press, Retrieved from
http://books.google.com.au/books?id=Gq6kAgAAQBAJ.
state-of-the-artinthecognitivemodelingandneuralmodelingof
Lewandowsky,S.,&Farrell,S.(2010).Computationalmodelingincognition:principles
the research field in question. Very tightly-linked models, with andpractice.Sage.
explicit and quantitative linking assumptions, are only possible Link, S. W., & Heath, R. A. (1975). A sequential theory of psychological
discrimination.Psychometrika,40,77–105.
in research fields with tractable quantitative models for both
Luce,R.D.(1986).Responsetimes.NewYork:OxfordUniversityPress.
behavioralandneuraldata. Nosofsky,R.M.(1986).Attention,similarity,andtheidentification–categorization
One of the more interesting implications of joint modeling is relationship.JournalofExperimentalPsychology:General,115,39–57.
Nosofsky,R.M.,&Palmeri,T.J.(1997).Anexemplar-basedrandomwalkmodelof
tryingtorelatetwostreamsofdatawithpotentiallyvastlydifferent
speededclassification.PsychologicalReview,104,266–300.
scales. In our particular example, we are combining RT data in Pouget, P., Logan, G. D., Palmeri, T. J., Boucher, L., Paré, M., & Schall, J. D.
seconds(from0to7),proportioncorrect(from0to1),andmean (2011). Neural basis of adaptive response time adjustment during saccade
amplitudes at Pz (90% fall in the range −7.5 to 11). Revealing countermanding.TheJournalofNeuroscience,31,12604–12612.
Provost, A., & Heathcote, A. (2015). Titrating decision processes in the mental
howthedifferentkindsofmodelmisfitinteract,andhowmuch rotationtask.PsychologicalReview,122,735–754.
influence they have relative to one another, is another of the Provost,A.,Johnson,B.,Karayanidis,F.,Brown,S.D.,&Heathcote,A.(2013).Two
strengths of the joint modeling approach we have used. Direct routestoexpertiseinmentalrotation.CognitiveScience,37,1321–1342.
Purcell,B.A.,Heitz,R.P.,Cohen,J.Y.,Schall,J.D.,Logan,G.D.,&Palmeri,T.J.(2010).
comparisonofthefitsofmodelthatvaryinjustonecomponent
Neurallyconstrainedmodelingofperceptualdecisionmaking.Psychological
can be very revealing about that component (conditional on Review,117,1113–1143.
the other model components being reasonable, of course). For Raaijmakers, J. G. W., & Shiffrin, R. M. (1981). Search of associative memory.
ν PsychologicalReview,88,93–134.
instance,ourmodels -ERPandt -ERPdifferedonlyontheneural
0 Rae, B., Heathcote, A., Donkin, C., Averell, L., & Brown, S. D. (2014). The hare
linkingcomponent,whereast 0 -ERPandBrev–t 0 -ERPdifferedonly andthetortoise:Emphasizingspeedcanchangetheevidenceusedtomake
on the behavioral component, and these pairings allowed us to decisions.JournalofExperimentalPsychology:Learning,Memory,andCognition,
40,1226–1243.
investigateinterestingpsychologicalquestions.
Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85,59–108.
Our example model demonstrates a very tight link between Riečanský,I.,&Jagla,F.(2008).Linkingperformancewithbrainpotentials:Mental
neuralandbehavioraldatainthefieldofsimpledecision-making. rotation-relatednegativityrevisited.Neuropsychologia,46,3069–3073.
Schall,J.D.(2001).Neuralbasisofdeciding,choosing,andacting.NatureReviews
Thisfieldhasseensomeexcellentinterdisciplinaryworkbetween
Neuroscience,2,33–42.
neuroscienceandpsychology (e.g.PurcellEtAl2010). Theexample Shadlen,M.N.,&Newsome,W.T.(1996).Motionperception:Seeinganddeciding.
model we developed shows how to adapt the LBA model for ProceedingsoftheNationalAcademyofSciences,93,628–633.
response time data to incorporate ERP data, recorded during Shepard,R.N.,&Metzler,J.(1971).Mentalrotationofthree-dimensionalobjects.
Science,171,701–703.
a mental rotation task. The result is a joint model that can Spiegelhalter,D.J.,Best,N.G.,Carlin,B.P.,&vanderLinde,A.(2002).Bayesian
simultaneously capture characteristics of data at the behavioral measuresofmodelcomplexityandfit.JournaloftheRoyalStatisticalSociety,
level(responsetimesandchoiceproportions)andtheneurallevel SeriesB,64,583–639.
Sutton,S.,Braren,M.,Zubin,J.,&John,E.(1965).Evoked-potentialcorrelatesof
(ERPs).Approacheslikethisareveryexciting,becausetheyhelpto stimulusuncertainty.Science,150,1187–1188.
reducebarriersbetweentwofieldsthathaveoperatedalongside Teller,D.(1984).Linkingpropositions.VisionResearch,24,1233–1246.
another rather than together for a long time. We hope that Trueblood,J.S.,Brown,S.D.,&Heathcote,A.(2014).Themultiattributelinear
ballistic accumulator model of context effects in multialternative choice.
modelingdataatdifferentlevelswithasinglesetofparameters
PsychologicalReview,121,179–205.
pavesthewaytoamoreintegrativecognitivescience. Turner, B. M., Forstmann, B. U., Love, B. C., Palmeri, T. J., & van Maanen, L.
(2017).Approachestoanalysisinmodel-basedcognitiveneuroscience.Journal
ofMathematicalPsychology,76,65–79.
References Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.-J.,Brown,S.D.,Sederberg,P.B.,&
Steyvers,M.(2013).ABayesianframeworkforsimultaneouslymodelingneural
Anderson,J.R.,&Fincham,J.M.(2014).Discoveringthesequentialstructureof andbehavioraldata.Neuroimage,72,193–206.
thought.CognitiveScience,38,322–352. Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013). A method
Atkinson,R.C.,&Shiffrin,R.M.(1968).Humanmemory:Aproposedsystemandits for efficiently sampling from distributions with correlated dimensions.
controlprocesses.PsychologyofLearningandMotivation,2,89–195. PsychologicalMethods,18,368–384.
Belliveau,J.,Kennedy,D.,McKinstry,R.,Buchbinder,B.,Weisskoff,R.,Cohen,M., Usher,M.,&McClelland,J.L.(2001).Onthetimecourseofperceptualchoice:The
et al. (1991). Functional mapping of the human visual cortex by magnetic leakycompetingaccumulatormodel.PsychologicalReview,108,550–592.
resonanceimaging.Science,254,716–719. vanRavenzwaaij,D.,Cassey,P.,&Brown,S.D.(2016).Asimpleintroductionto
Brenner,D.,Williamson,S.,&Kaufman,L.(1975).Visuallyevokedmagneticfields MarkovchainMonte-Carlo.PsychonomicBulletin&Review,inpress.
ofthehumanbrain.Science,190,480–482. Wagenmakers,E.-J.(2009).Methodologicalandempiricaldevelopmentsforthe
Brown,S.,&Heathcote,A.(2008).Thesimplestcompletemodelofchoicereaction Ratcliffdiffusionmodelofresponsetimesandaccuracy.EuropeanJournalof
time:Linearballisticaccumulation.CognitivePsychology,57,153–178. CognitivePsychology,21,641–671.