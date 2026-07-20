JournalofMathematicalPsychology76(2017)142–155
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
On the efficiency of neurally-informed cognitive models to identify
latent cognitive states✩
GuyE.Hawkinsa,∗ ,MatthiasMittnerb,BirteU.Forstmanna,AndrewHeathcotec
aAmsterdamBrainandCognitionCenter,UniversityofAmsterdam,Amsterdam,TheNetherlands
bDepartmentofPsychology,UniversityofTromsø,Tromsø,Norway
cSchoolofMedicine–DivisionofPsychology,UniversityofTasmania,Hobart,Tasmania,Australia
h i g h l i g h t s
• Explorestherecoveryofcognitivemodelsthatareinformedwithneuraldata.
• Contraststwoframeworksforusingneuraldatatoidentifylatentcognitivestates.
• Neuraldatahavemorepowertorecoverdiscreteversuscontinuouslatentstates.
• Reliablyidentifyinglatentcognitivestatesdependsoneffectsizeinneuraldata.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Psychologicaltheoryisadvancedthroughempiricaltestsofpredictionsderivedfromquantitativecog-
Availableonline25July2016 nitivemodels.Ascognitivemodelsaredevelopedandextended,theytendtoincreaseincomplexity–
leadingtomoreprecisepredictions–whichplacesconcomitantdemandsonthebehavioraldataused
Keywords:
todiscriminatebetweencandidatetheories.Toaiddiscriminationbetweencognitivemodelsand,more
Cognitivemodel
recently,toconstrainparameterestimation,neuraldatahavebeenusedasanadjuncttobehavioraldata,
Behavioraldata
orasacentralstreamofinformation,intheevaluationofcognitivemodels.Suchamodel-basedneuro-
Neuraldata
Modelrecovery scienceapproachentailsmanyadvantages,includingprecisetestsofhypothesesaboutbrain–behavior
Simulation relationships.Therehave,however,beenfewsystematicinvestigationsofthecapacityforneuraldatato
constraintherecoveryofcognitivemodels.Throughthelensofcognitivemodelsofspeededdecision-
making,weinvestigatedtheefficiencyofneuraldatatoaididentificationoflatentcognitivestatesin
modelsfittobehavioraldata.Westudiedtwotheoreticalframeworksthatdifferedintheirassumptions
aboutthecompositionofthelatentgeneratingstate.Thefirstassumedthatobservedperformancewas
generatedfromamixtureofdiscretelatentstates.Thesecondconceivedofthelatentstateasdynami-
callyvaryingalongacontinuousdimension.Weusedasimulation-basedapproachtocomparerecovery
oflatentdata-generatingstatesinneurally-informedversusneurally-uninformedcognitivemodels.We
foundthatneurally-informedcognitivemodelsweremorereliablyrecoveredunderadiscretestaterep-
resentationthanacontinuousdimensionrepresentationformediumeffectsizes,althoughrecoverywas
difficultforsmallsamplesizesandmoderatenoiseinneuraldata.Recoveryimprovedforbothrepresen-
tationswhenalargereffectsizedifferentiatedthelatentstates.Weconcludethatneuraldataaidsthe
identificationoflatentstatesincognitivemodels,butdifferentframeworksforquantitativelyinform-
ingcognitivemodelswithneuralinformationhavedifferentmodelrecoveryefficiencies.Weprovidefull
workedexamplesandfreely-availablecodetoimplementthetwotheoreticalframeworks.
©2016ElsevierInc.Allrightsreserved.
1. Introduction
✩ This research was supported by a Netherlands Organisation for Scientific Quantitative models that explicate the cognitive processes
Research(NWO)Vidi(452-11-008)granttoBirteForstmannandanAustralian driving observed behavior are becoming increasingly complex,
Research Council (ARC) (DP110100234) Professorial Fellowship to Andrew
Heathcote.Theauthorsdeclarenocompetingfinancialinterests.
∗
Correspondence to: Amsterdam Brain and Cognition Center, University of
Amsterdam,NieuweAchtergracht129,Amsterdam1018WS,TheNetherlands. E-mailaddress:guy.e.hawkins@gmail.com(G.E.Hawkins).
http://dx.doi.org/10.1016/j.jmp.2016.06.007
0022-2496/©2016ElsevierInc.Allrightsreserved.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 143
leadingtofiner-grainedpredictionsfordata.Althoughincreasingly see Turner, Forstmann et al., 2013). To this end, we condition
precisemodelpredictionsareundoubtedlyabenefitforthefield, on neural data; that is, we do not consider generative models
they also increase the demands placed on data to discriminate ofneuraldata.Rather,weexploretractableandsimplemethods
betweencompetingmodels.Thepredictionsofcognitivemodels thataugmentcognitivemodelsusingneuraldataascovariatesin
have traditionally been tested against behavioral data, which is ordertogaingreaterinsightintocognitionthanispossiblethrough
typicallylimitedtochoicesand/orresponsetimes.Suchbehavioral considerationofbehavioraldatainisolation.
datahavebeenextremelyusefulindiscriminatingbetweenmodel Throughoutthemanuscript,wepositionourworkwithinthe
architectures (e.g., Anderson et al., 2004; Brown & Heathcote, theoreticalcontextofmindwandering.Overthepastdecade,the
2008; Forstmann, Ratcliff, & Wagenmakers, 2016; Nosofsky & scientificstudyofmindwanderinghasreceivedgreatinterestfrom
Palmeri, 1997; Ratcliff & Smith, 2004; Shiffrin & Steyvers, 1997; behavioral(e.g.,Bastian&Sackur,2013;Cheyne,Solman,Carriere,
Tversky & Kahneman, 1992). As model predictions increase in &Smilek,2009)andneural(e.g.,Andrews-Hanna,Reidler,Sepul-
precision, however, we approach a point where behavioral data cre,Poulin,&Buckner,2010;Christoff,Gordon,Smallwood,Smith,
have limited resolution to further constrain and discriminate &Schooler,2009;Weissman,Roberts,Visscher,&Woldorff,2006)
betweentheprocessesassumedbythemodelsofinterest. perspectives, though there have been few attempts to integrate
Theproblemofbehavioraldataprovidinglimitedconstraintis thetwostreamsofinformationinamodel-basedcognitiveneu-
compoundedwhenoneaimstostudynon-stationarity.Cognitive roscienceframework(foranexception,seeMittneretal.,2014).
modelstypicallyassumeastationarygenerativeprocesswhereby Thestudyofmindwanderingisparticularlyrelevanttoouraim
trialswithinanexperimentalconditionaretreatedasindependent of identifying latent cognitive states as it is a phenomenon that
and identically distributed random samples from a probabilistic hasbeenstudiedundervarious,qualitativelydistinct,hypotheses
model with a specified set of parameters. This assumption has abouthowlatentstatesgiverisetoobservedperformance(Small-
provenextremelyuseful,bothpracticallyandtheoretically,butis wood & Schooler, 2006, 2015), which we expand upon below.
notsupportedbyfine-grainedempiricalanalysis(e.g.,Craigmile, Mindwandering,therefore,servesasanexcellentvehiclethrough
Peruggia, & Van Zandt, 2010; Wagenmakers, Farrell, & Ratcliff, which to demonstrate our methodological approach. Our work-
2004).Recentworkinthestudyofstimulus-independentthought, inghypothesisisthatmindwanderingisaneuralstateorprocess
or mind wandering, provides a psychological mechanism that thataffectstheparametersofcognitivemodels,whichinturnaf-
can explain these findings, at least in part, in terms of observed fectobservedbehavioralperformance(Hawkins,Mittner,Boekel,
performance arising from two or more latent data-generating Heathcote,&Forstmann,2015).Ourapproachinvertsthischainof
states.Oneprominenttheoryproposesthatongoingperformance causation:wefitbehavioraldatawithcognitivemodelsthatarein-
is driven by two distinct phases: perceptual coupling – where formedwithneuraldata,andcomparetheirfittocognitivemodels
attentional processes are directed to incoming sensory input that are not informed with neural data. This allows us to assess
and completing the ongoing task – and perceptual decoupling whatcanbelearntaboutmindwanderinginawaythatisnotfea-
– where attention is diverted from sensory information toward siblewithoutthediscriminativepoweroftheneuraldata.
inner thoughts (for detailed review, see Smallwood & Schooler, Through the lens of cognitive models of speeded decision-
2015).Theperceptualdecouplinghypothesisofmindwandering making, we consider two approaches that use neural data to
proposes,therefore,thatobservedbehavioristheendresultofa constrain cognitive models, which in turn helps to identify
mixtureofdiscretelatentdata-generatingstates.Togaininsight both when people mind wander and the effect it has on task
into the processes underlying the phases of perceptual coupling performance.Wenote,however,thatourmethodsgeneralizeto
and decoupling, the goal of the cognitive modeler is to use the anydomainofstudythatutilizesneuraldata–oranyadditional
available data to determine the optimal partition of trials into streamofdata,forthatmatter–toaididentificationoflatentdata-
latentstates. generating states and fit the behavioral data arising from those
On the basis of behavioral data alone, such as choices and stateswithcognitivemodels.
response times, reliably identifying discrete latent states can be We consider two general approaches to incorporating mind
difficult or near impossible. In an example of this approach, wandering within a modeling framework. The first approach
Vandekerckhove, Tuerlinckx, and Lee (2008) aimed to identify assumesthatobservedbehaviorarisesfromamixtureofdiscrete
contaminant trials – data points not generated by the process latentstates,whichmayhavepartiallyoverlappingoruniquesets
of interest – in a perceptual decision-making experiment. They ofdata-generatingparameters.WerefertothisastheDiscreteState
defined a latent mixture model in a Bayesian framework that Representation. One might think of the latent states as reflecting
attemptedtopartitiontrialsthatweresampledfromthe(diffusion an on-task state, where attention is directed to external stimuli,
model) process of interest from contaminant trials distributed ortask-relatedthoughts,andanoff-taskstate,whereattentionis
accordingtosomeotherprocess.Inattemptingtosegmenttrials directed to internal stimuli, or task-unrelated thoughts, similar
to latent classes, the diffusion model was only informed by the totheperceptualdecouplinghypothesis(Smallwood&Schooler,
same choice and response time data it was designed to fit. For 2015). Alternatively, the latent states might reflect executive
a representative participant, only 0.6% of their 8000 trials were control, where an executive system oversees maintenance of
classifiedascontaminants,indicatingeitheraremarkableability goal-directedbehavior,andexecutivefailure,whichoccurswhen
of the participant to remain on task (which is unlikely; see, the executive control system fails to inhibit automatically cued
e.g.,Killingsworth&Gilbert,2010),or,morelikely,tothelimited internal thoughts that derail goal-directed behavior (McVay &
abilityofbehavioraldataalonetosegmenttrialsintolatentstates. Kane,2010).Regardlessofthelabelsassignedtothelatentstates,
Ratherthanrelyingsolelyonbehavioraldata,hereweexamine models assuming a discrete state representation aim to first
whetheraugmentingcognitivemodelswithanadditionalstream identify the mutually exclusive latent states and then estimate
ofinformation–suchasneuraldata,whetherthatinvolvessingle partiallyoverlappingordistinctsetsofmodelparametersforthe
cell recordings, EEG, MEG, or fMRI – aids identification of latent discretestates(forasimilarapproach,seeMittneretal.,2014).We
data-generatingstatesunderlyingobservedbehavior.Ouraimis notethatadiscretestaterepresentationisalsoconsideredoutside
toinvestigatewhethertheadditionofneuraldatacanimproveour thecontextofmindwandering.Forexample,BorstandAnderson
accountofthebehavioraldata,andinparticulartheidentification (2015) developed a hidden semi-Markov model approach that
of latent states, rather than accounting for the joint distribution usedacontinuousstreamofEEGdatatoidentifydiscretestages
of behavioral and neural data (for joint modeling approaches, ofprocessinginassociativeretrieval.

144 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
Thesecondapproachgeneralizesthediscretestaterepresenta-
tion,relaxingtheassumptionthatlatentstatesaremutuallyex-
clusive.Thisapproachassumesadynamicallyvaryinglatentstate
where,forexample,atalltimesaparticipantwillfallatsomepoint
| along a continuum |                 | that spans | from      | a completely |                 | on-task   | focus  |     |     |     |     |     |     |     |     |
| ----------------- | --------------- | ---------- | --------- | ------------ | --------------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| through           | to a completely | off-task   |           | focus. We    | refer           | to this   | second |     |     |     |     |     |     |     |     |
| approach          | as the          | Continuous | Dimension |              | Representation, |           | and it |     |     |     |     |     |     |     |     |
| approximates      | ‘executive      |            | resource’ | theories     | of mind         | wandering |        |     |     |     |     |     |     |     |     |
(e.g.,Smallwood&Schooler,2006;Teasdaleetal.,1995).Thisclass
oftheoriesstatesthatexecutiveresourcesarerequiredtoperform
goal-directedtasks.Thepoolofresourcesisfinite,andcompeting
demands,suchasmindwanderingfromthetaskathand,reduce
| the resources | available | to  | complete | the primary |     | task, leading | to  |     |     |     |     |     |     |     |     |
| ------------- | --------- | --- | -------- | ----------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
suboptimaltaskperformance.Theresourcesavailabletocomplete
ataskcaneffectivelybeconsideredacontinuousvariable:attimes
therearemoreresourcesavailabletocompletethetaskthanoth-
ers,andthiscanvaryinpotentiallycomplexwaysfromonetrialto
thenext.Modelsassumingacontinuousdimensionrepresentation
aimtoregresssingle-trialmeasuresofneuralactivityontostruc- Fig. 1. Schematic representation of the diffusion model of speeded-decision
turedtrial-by-trialvariationinmodelparameters(forsimilarap- making.ReproducedwithpermissionfromHawkinsetal.(2015).
proaches,seeCavanaghetal.,2011;Franketal.,2015;Nunez,Srini-
vasan,&Vandekerckhove,2015;Nunez,Vandekerckhove,&Srini-
tworesponsealternativesuntilitcrossesanabsorbingboundary–
vasan,2017).Totheextentthatthesingle-trialregressorsindex
apre-determinedthresholdamountofevidence–whichtriggersa
thelatentconstructofinterest,thisapproachdynamicallytracks
response.Thepredictedchoiceisdeterminedbytheboundarythat
theeffectofneuralfluctuationsonchangesinmodelparameters.
wascrossed,andthepredictedresponsetimeisthetimetakenfor
| We use | a simulation-based |     |     | approach | to explore | how | well |     |     |     |     |     |     |     |     |
| ------ | ------------------ | --- | --- | -------- | ---------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
theprocesstoreachtheboundaryplusafixedoffsettimetoac-
neuraldataconstrainstheidentificationofdata-generatingstates countforprocessessuchasencodingthestimulusandproducing
whenfittingcognitivemodelstobehavioraldata.Wefirstsimulate amotorresponse(e.g.,abuttonpress).
data from models that assume a non-stationary data-generating Fig. 1 provides a schematic overview of a choice between
process(i.e.,alatentcognitivestatethatchangesthroughoutthe
leftwardandrightwardmotioninthediffusiondecisionmodel.The
courseofanexperiment).Wethenfitmodelstothesyntheticdata
modelhasfourcoreprocessingparameters:thestartingpointof
thatvaryintheirknowledgeofthelatentdata-generatingstates:
evidenceaccumulation,whichcanimplementbiasestowardone
somemodelscompletelyignorethepresenceofalatentmixture
|     |     |     |     |     |     |     |     | response | or another | (z); | the | average | rate at | which information |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---- | --- | ------- | ------- | ----------------- | --- |
in data (i.e., they are misspecified), and others assume partial v
|     |     |     |     |     |     |     |     | is extracted | from | the stimulus, |     | known | as the | drift rate | ( ), the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ------------- | --- | ----- | ------ | ---------- | -------- |
throughtoperfectknowledgeofthelatentdata-generatingstates.
|            |               |           |        |              |         |     |           | amount       | of evidence | required    |       | for a response, |          | which      | represents |
| ---------- | ------------- | --------- | ------ | ------------ | ------- | --- | --------- | ------------ | ----------- | ----------- | ----- | --------------- | -------- | ---------- | ---------- |
| The degree | of partial    | knowledge |        | about latent | states  | is  | assumed   |              |             |             |       |                 |          |            |            |
|            |               |           |        |              |         |     |           | cautiousness | in          | responding, | known | as              | boundary | separation | (a);       |
| to reflect | the precision | of        | neural | data that    | informs | the | analysis. |              |             |             |       |                 |          |            |            |
andthetimerequiredforelementsoutsidethedecisionprocess,
| When a | neural measure | or  | measures | are | perfectly | predictive | of  |          |              |     |      |                 |     |                 |     |
| ------ | -------------- | --- | -------- | --- | --------- | ---------- | --- | -------- | ------------ | --- | ---- | --------------- | --- | --------------- | --- |
|        |                |     |          |     |           |            |     | known as | non-decision |     | time | (T er ). Modern |     | implementations | of  |
thelatentgeneratingstates,thepartitionofbehavioraldatatoone
|              |            |         |     |                 |     |          |     | the diffusion | model | assume |     | trial-to-trial | variability |     | in some |
| ------------ | ---------- | ------- | --- | --------------- | --- | -------- | --- | ------------- | ----- | ------ | --- | -------------- | ----------- | --- | ------- |
| latent state | or another | mirrors | the | data-generating |     | process, | and |               |       |        |     |                |             |     |         |
modelparameterstoreflecttheassumptionthatperformancehas
themodelthatassumesamixtureoflatentgeneratingstateswill
systematicandnonsystematiccomponentsoverthecourseofan
bepreferredoverthe(misspecified)modelthatmarginalizesover
experiment(Ratcliff&Tuerlinckx,2002).Theseparametersinclude
latentstates.Asthestrengthoftherelationshipbetweentheneural
|     |     |     |     |     |     |     |     | the drift | rate, starting | point, | and | non-decision |     | time. | Specifically, |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------------- | ------ | --- | ------------ | --- | ----- | ------------- |
measureandthepartitioninbehavioraldataweakens,weought
|     |     |     |     |     |     |     |     | on trial | i the drift | rate | is sampled | from | a Gaussian |     | distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ---- | ---------- | ---- | ---------- | --- | ------------ |
toobtainlessevidenceforthemodelthatassumesamixtureof
|                                                        |     |     |     |     |     |     |     | with mean | v and | standard | deviation |     | η , v ∼ | N (v,η) | ; the start |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --------- | ----- | -------- | --------- | --- | ------- | ------- | ----------- |
| latentstatesindata.Ourprimaryaimistodeterminetheamount |     |     |     |     |     |     |     |           |       |          |           |     | i       |         |             |
∼
ofnoisethatcanbetoleratedintherelationshipbetweenneural pointissampledfromauniformdistributionwithrangesz,z i
|     |     |     |     |     |     |     |     | ( − sz, | + sz) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- |
andbehavioraldatabeforethemisspecifiedmodelthatcollapses U z z ;andthenon-decisiontimeissampledfroma
|     |     |     |     |     |     |     |     | 2   | 2   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
acrossthe(true)latentstatesispreferred.Ouroutcomemeasure uniformdistributionwithrangest,T er,i ∼U ( T − s t, T + s t) .
|     |     |     |     |     |     |     |     |     |     |     |     |     |     | er 2 | er 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- |
Inallcases,wesimulateddatafromahypotheticalexperiment
ofinterestis,therefore,theprobabilitywithwhichweselectthe
|     |     |     |     |     |     |     |     | of a two-alternative |     | forced | choice | task | with | a single | condition. |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | ------ | ------ | ---- | ---- | -------- | ---------- |
modelthatassumesmorethanonelatentgeneratingstateindata,
whichwasthetruedata-generatingmodelinallcases. The use of a single experimental condition mirrors almost all
laboratory-basedstudiesofmindwandering,whichtendtofocus
1.1. Diffusionmodelofspeededdecision-making on vigilance tasks such as the sustained attention to respond
task(SART;Robertson,Manly,Andrade,Baddeley,&Yiend,1997;
In all simulations, we studied sequential sampling models of Smallwood & Schooler, 2006; Smilek, Carriere, & Cheyne, 2010).
|                  |            |            |           |               |            |     |           | The SART | is typically    | implemented |       | as      | a single-condition |          | go/no-      |
| ---------------- | ---------- | ---------- | --------- | ------------- | ---------- | --- | --------- | -------- | --------------- | ----------- | ----- | ------- | ------------------ | -------- | ----------- |
| decision-making, |            | and the    | diffusion | model         | of speeded |     | decision- |          |                 |             |       |         |                    |          |             |
|                  |            |            |           |               |            |     |           | go task  | with infrequent |             | no-go | stimuli | (i.e.,             | stimuli  | requiring a |
| making in        | particular | (Forstmann |           | et al., 2016; | Ratcliff   | &   | McKoon,   |          |                 |             |       |         |                    |          |             |
|                  |            |            |           |               |            |     |           | response | to be           | withheld),  | with  | the     | aim of             | inducing | boredom     |
2008;Smith&Ratcliff,2004).Thediffusionmodel,aswithmostse-
quentialsamplingmodels,assumesthatsimpledecisionsaremade andhencemindwandering.Thesequentialsamplingmodelswe
through a gradual process of accumulating sensory information study here can be generalized to experimental paradigms with
fromtheenvironment.Thesensoryinformationinfluencesanevi- partialresponsetimedata–suchasgo/no-goandstop-signaltasks
dencecounterthattrackssupportforoneresponsealternativeover (Gomez,Ratcliff,&Perea,2007;Logan,VanZandt,Verbruggen,&
another;forexample,whetheramotionstimulusmovestotheleft Wagenmakers,2014)–sotheresultsreportedherearerelevant
orrightofadisplay,orwhetherastringoflettersrepresentsaword tothetasksandexperimentalparadigmstypicallystudiedinthe
ornot.Theevidencecountercontinuestotrackevidenceforthe mindwanderingliterature.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 145
a
c
b
Fig.2. Anexemplarysyntheticdatasetgeneratedfromtheon-taskandoff-taskstatesofthedual-statemodel(panelsaandb),andthefitofthesingle-statemodeltothe
samedataset,collapsedoverlatentstates(panelc).Responsetimedistributionsforcorrectresponsesareshowntotherightofzeroanddistributionsforerrorresponsesare
showntotheleftofzero(i.e.,mirroredaroundthezero-pointonthex-axis).Greenandredlinesshowcorrectanderrorresponses,respectively,fromtheposteriorpredictive
distributionofthesingle-statemodel(panelc).Theprobabilityofacorrectresponseinsyntheticdataisdenotedp,andthecorrespondingpredictedprobabilityfromthe
single-statemodelisdenotedpˆ(panelc).(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
Our primary aim was to identify the latent data-generating thedriftratefortheon-taskstatewaslargerthanthedriftratefor
states in data. This is a question pertinent to the individual- theoff-taskstate,whichimpliesthatmindwanderingreducesthe
participant level – when was the participant on-task, and when efficiencyofinformationprocessing.Thisassumptionisconsistent
weretheyoff-task–thuswesimulateandfitmodelstodataatthe with empirical results suggesting that mind wandering leads to
individual-participantlevel. slower and more variable response times with a greater error
rate (e.g., Bastian & Sackur, 2013; Cheyne et al., 2009), which
is qualitatively similar to the effect of a reduction in drift rate.
2. Discretestaterepresentation
Specifically,wesetthedriftratefortheon-taskstateto v = 2
on
andtheoff-taskstateto v = 1.Allotherparameterswereset
2.1. Generatingsyntheticdata tothefollowingvalues,fo o r ff bothstates:a = 1,z = 0 . 5(i.e.,no
responsebias),T =0 . 15s, η=1,andthetrial-to-trialvariability
Syntheticdataweregeneratedfromthediscretestaterepresen- er
parametersforthestartpointofevidenceaccumulationandnon-
tationbyassumingthat80%oftrialswerefromtheon-taskstate
decisiontimewerebothsetto0.Thediffusioncoefficientwasfixed
andtheremaining20%oftrialswerefromtheoff-taskstate.One tos=1inallsyntheticdataandmodelfitswereobtainedusingthe
couldmanipulatetheratioofon-tasktooff-tasktrialsasaparam-
‘rtdists’packagefortheRprogrammingenvironment(Singmann,
eterofthemodelrecoveryexercise.Wechoseinsteadtoselecta
Brown,Gretton,&Heathcote,2016).Anexemplarysyntheticdata
fixedvaluethatmightbeconsideredaconservativeestimateofre-
set is shown in Fig. 2(a) and (b). The synthetic data of the on-
portedratesofmindwanderinginexperimentaltasksthatmirror taskstatedifferedtotheoff-taskstateintermsofhigheraccuracy
thesetupofoursimulatedexperiment,soastonotoverstatethe and faster mean response times that were less variable. These
estimatedpowerofourresults(e.g.,somehavereportedthatmind differencesindicatethattherewasareliablesignalinbehavioral
wanderingoccursbetween30%–50%ofthetime;Killingsworth& datathatdifferentiatedthelatentstates.
Gilbert,2010).1
We generated synthetic data across a wide range of sample
Ingeneratingsyntheticdata,weconstrainedtheparametersof sizes(i.e.,numberoftrialscompletedbyasyntheticparticipant).
theon-taskandoff-taskstatestoidenticalvalues,exceptforthe Ourmotivationwastodeterminetheefficiencyofneuraldatato
magnitudeofthedriftrate.Wemadetheplausibleassumptionthat identifydiscretelatentstatesusingsamplesizesconsideredvery
small for fitting sequential sampling models to data, through to
an approximate asymptotic limit with very large sample sizes.
Specifically, we simulated 200 synthetic data sets from each of
1 Nevertheless,toassureourselvesthatourresultswerenotdependentonthe
sample sizes 100, 250, 500, 1000, 2000, 5000, and 10000 trials.
ratioofon-tasktooff-tasktrialsandtheparametersettingsdescribedbelow,we
Therefore,forsamplesizesof100trials,forexample,therewere
conductedaparallelanalysiswheresyntheticdataweregeneratedfromadiscrete
staterepresentationwithanequalratioofon-tasktooff-tasktrialsandalower 80‘on-task’and20‘off-task’trials,andfor10000trialstherewere
driftratefortheon-taskstate(v on =1.8).Following(4)and(5),thesesettingsgive 8000‘on-task’and2000‘off-task’trials.
anequivalenteffectsizetothatreportedintheprimarysimulation.Allresultsof
theparallelanalysismirrorthoseshownintheleftpanelofFig.3.Combinedwith 2.2. Modelspecification
theresultsshowninFig.4,thisfindingsuggeststhattheprimaryfactorinfluencing
recoveryofthetruelatentgeneratingstateisthesizeoftheeffectthattheneural
dataexertonthelatentstate,andnotparticulardata-generatingparametersettings Wefittwotypesofdiffusionmodelstoeachsyntheticdataset:
ofthecognitivemodel. asingle-stateandadual-statemodel.IntheAppendix,weoutline

146 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
thestepsinvolvedinperformingananalysisassumingadiscrete thateverytrialthatwasgeneratedfromtheon-taskstatehad0.8
state representation and provide accompanying R code (R Core probabilityofbeingcorrectlyassignedtotheon-taskstateinthe
Team,2016)thatusesthertdistspackage(Singmannetal.,2016). fitted model, and 0.2 probability of incorrect assignment to the
off-taskstateinthefittedmodel.Thereversewasalsoassumed:
|     |     |     |     |     |     |     |     | trials generated |     | from the | off-task | state | had | 0.8 | probability | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | -------- | -------- | ----- | --- | --- | ----------- | --- |
2.2.1. Single-statemodel
|                  |     |       |      |              |       |     |           | assignment | to the | off-task | state | in  | the fitted | model, | and | 0.2 |
| ---------------- | --- | ----- | ---- | ------------ | ----- | --- | --------- | ---------- | ------ | -------- | ----- | --- | ---------- | ------ | --- | --- |
| The single-state |     | model | is a | misspecified | model | in  | the sense |            |        |          |       |     |            |        |     |     |
probabilityofassignmenttotheon-taskstate.Thisvaluemimics
thatitmarginalizes(collapses)overtrialsgeneratedfromtheon-
|     |     |     |     |     |     |     |     | the classification |     | accuracy | achieved | in  | Mittner | et al. | (2014). | We  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | -------- | -------- | --- | ------- | ------ | ------- | --- |
taskandoff-tasklatentstates;thisapproachisequivalenttonot
|     |     |     |     |     |     |     |     | exploredarangefromp |     |     |     | = 0 . 5(theneuraldataprovideno |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- |
using any neural data to inform cognitive modeling. The single- correct
informationaboutthelatentstate,sotrialsarerandomlyallocated
statemodelingisrepresentativeofthedominantapproachinthe totheon-oroff-taskstate)throughtop =1(theneuraldata
correct
literaturethatgenerallymakesnoattempttoaccountforpotential provideperfectknowledgeofthegeneratingstate),inincrements
task-unrelated thoughts and their effects on task performance. of0.05.Therefore,foreachsyntheticdataset,wecomparedthefit
Thesingle-statemodelfreelyestimatedthefollowingparameters ofthesingle-statemodelto11dual-statemodelscorrespondingto
fromdata:startpoint(z),trial-to-trialvariabilityinstartpoint(sz), therangeinp .Foreachvalueofp ,wedeterminedwhich
|     |     |     |     | v   |     |     |     |     | correct |     |     |     | correct |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | ------- | --- | --- | --- |
boundary separation (a), drift rate ( ), trial-to-trial variability in model (single state, dual state) provided the most parsimonious
η
driftrate( ),andnon-decisiontime(T ).Trial-to-trialvariability accountofthesyntheticdataset.
er
| innon-decisiontimewasfixedtost |     |          |      | = 0.Wemadethisdecision |            |     |         |     |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | -------- | ---- | ---------------------- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| as we deemed                   | it  | unlikely | that | the parameter          | estimation |     | routine |     |     |     |     |     |     |     |     |     |
2.3. Parameterestimation
| would compensate |          | for | the misspecification |     | of         | the single-state |     |            |     |      |           |           |     |              |     |     |
| ---------------- | -------- | --- | -------------------- | --- | ---------- | ---------------- | --- | ---------- | --- | ---- | --------- | --------- | --- | ------------ | --- | --- |
| model with       | a change | in  | the parameter        |     | reflecting | non-decision     |     |            |     |      |           |           |     |              |     |     |
|                  |          |     |                      |     |            |                  |     | We sampled |     | from | the joint | posterior |     | distribution | of  | the |
timevariability,andourBayesianparameterestimationroutines
|     |     |     |     |     |     |     |     | parameters | of each | model | using | differential |     | evolution | Markov |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------- | ----- | ----- | ------------ | --- | --------- | ------ | --- |
werecomputationallymuchmorefeasiblewithoutthenumerical
chainMonteCarlo(Turner,Sederberg,Brown,&Steyvers,2013).
integrationrequiredforestimationofthestparameter. We assumed prior distributions that had a considerable range
around,butconveyedrelativelylittleinformationabout,thetrue
| 2.2.2. Dual-statemodel |     |     |     |     |     |     |     | data-generatingparametervalues: |     |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
The dual-state model acknowledged the on-task and off-task v[single-state]∼ ( , ,−5 , ),
|               |           |           |     |                      |                 |           |          |           |              | N 0      | 2     | 5     |     |     |     |     |
| ------------- | --------- | --------- | --- | -------------------- | --------------- | --------- | -------- | --------- | ------------ | -------- | ----- | ----- | --- | --- | --- | --- |
| generating    | states    | in data,  | by  | allowing             | for differences |           | in drift |           |              |          |       |       |     |     |     |     |
|               |           |           |     |                      |                 |           |          | v , v     | [dual-state] | ∼        | ( ,   | ,−5 , | ),  |     |     |     |
|               |           |           |     |                      |                 |           |          | on off    |              |          | N 0 2 | 5     |     |     |     |     |
| rate between  | trials    | allocated | to  | the on-task          | and             | off-task  | states   |           |              |          |       |       |     |     |     |     |
|               |           | v         | v   |                      |                 |           |          | a , s v ∼ | N ( 1 , 1 ,  | 0 , 2 ), |       |       |     |     |     |     |
| (i.e., freely | estimated | on        | and | off , respectively). |                 | All other | model    |           |              |          |       |       |     |     |     |     |
|               |           |           |     |                      |                 |           |          | , ,       |              | ( , ),   |       |       |     |     |     |     |
parameters were constrained to be equal across the two states z sz T ∼ Beta 1 1
|     |     |     |     | =   |     |     |     | er  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(asinthesingle-statemodel,st 0wasfixedeverywhere).The (µ,σ, , ) µ
|                                                          |     |     |     |     |     |     |     | whereN   | a         | b denotesaNormaldistributionwithmean |     |      |       |          |       | ,     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ------------------------------------ | --- | ---- | ----- | -------- | ----- | ----- |
| dual-statemodel,therefore,assumedsomeknowledgeofthedata- |     |     |     |     |     |     |     |          |           | σ                                    |     |      |       |          |       |       |
|                                                          |     |     |     |     |     |     |     | standard | deviation | , truncated                          |     | to a | lower | limit of | a and | upper |
generating structure in that there were two states that differed (α,β)
|     |     |     |     |     |     |     |     | limitofb,andBeta |     |     | denotesaBetadistributionwithshape |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- |
onlyindriftrate.Ourresultscanthusbeinterpretedasa‘bestcase’ α β
|     |     |     |     |     |     |     |     | parameters | and | . Parameters |     | z and | sz were | estimated |     | as a |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------ | --- | ----- | ------- | --------- | --- | ---- |
scenario;additionalmisspecificationinfreeparametersacrossthe
proportionofparametera,andhencewereconstrainedtotheunit
| discrete states, | or  | in the | number | of discrete | states, | may | worsen |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------ | ------ | ----------- | ------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
interval.
modelrecoveryrelativetothesingle-statemodel.
|     |     |     |     |     |     |     |     | Independently |     | for all | models, | we  | initialized | 18  | chains | with |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------- | ------- | --- | ----------- | --- | ------ | ---- |
Wedid,however,introducemisspecificationtothedual-state
|     |     |     |     |     |     |     |     | random | samples | from | the prior | distribution. |     | Chains | were | first |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ---- | --------- | ------------- | --- | ------ | ---- | ----- |
modelintermsofthereliabilitywithwhichtrialswereallocated
|     |     |     |     |     |     |     |     | run for 250 | iterations | with | the | differential | evolution |     | probability |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | ---- | --- | ------------ | --------- | --- | ----------- | --- |
tothetruegeneratingstate.Thatis,wesystematicallymanipulated
|     |     |     |     |     |     |     |     | of migration | set | to 0.05. | Once | initialization |     | was | complete, | the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | ---- | -------------- | --- | --- | --------- | --- |
theprobabilitythattrialsgeneratedfromtheon-taskstatewere
|     |     |     |     |     |     |     |     | migration | probability | was | set | to zero | and we | sampled | from | the |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ----------- | --- | --- | ------- | ------ | ------- | ---- | --- |
in the set of trials allocated to the on-task state in the fitted joint posterior distribution of the parameters in phases of 1000
model, and similarly for the off-task state. In the sense that the iterations.Aftereachphasewecheckedchainconvergenceusing
ˆ
setoftrialsgeneratedfromtheon-taskstatewasnotnecessarily themultivariatepotentialscalereductionfactor(R statistic;Brooks
the same set of trials fitted as the ‘on-task’ state, this model is & Gelman, 1998), using a criterion of R ˆ < 115 . to indicate
misspecified.Werefertothisformofmisspecificationasstate-level convergence(visualinspectionofasampleofchainssupportedthis
misspecification,whichisdistinctfromparametermisspecification conclusion).2 If the chains had converged after a phase of 1000
(i.e., allowing the wrong parameters to vary with state). State- iterations, the parameter estimation routine was terminated. If
levelmisspecificationmimicsthecapacityforanexternalstream not, another 1000 iterations were started from the end point of
ofinformation,suchasaneuraldata,toreliablypartitiontrialsinto theprevious1000iterations,andtheprocedurerepeateduntilthe
thetrue(data-generating)latentstate.Forexample,Mittneretal. chainshadconverged.
(2014)trainedasupportvectormachinetousearangeoffMRIand
pupilmeasurementstoclassifytrialsfromastop-signalparadigm
2.4. Modelselection
| to on-task | or off-task | states. | Their | classifier | achieved |     | expected |     |     |     |     |     |     |     |     |     |
| ---------- | ----------- | ------- | ----- | ---------- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
accuracy of 79.7% (relative to self-reported mind-wandering), ModelselectionwasperformedwiththeDevianceInformation
Criterion(DIC;Spiegelhalter,Best,Carlin,&vanderLinde,2002),3
| implying | that they   | could  | expect  | to correctly |     | classify | four out |     |     |     |     |     |     |     |     |     |
| -------- | ----------- | ------ | ------- | ------------ | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| of every | five trials | to the | on-task | or off-task  |     | states,  | assuming |     |     |     |     |     |     |     |     |     |
therewasatruedistinctioninthetwolatentstatesinthedata-
| generatingprocess. |     |     |     |     |     |     |     | 2           |             |     |           |              |     | ˆ        | ˆ < 1.1) |      |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | --------- | ------------ | --- | -------- | -------- | ---- |
|                    |     |     |     |     |     |     |     | Preliminary | simulations |     | indicated | lower values | of  | R (e.g., | R        | were |
Although it is likely that our simulated neural data leads producedbylongerseries,butwithoutanychangeinconclusions;wechosealength
to better-than-chance classification accuracy, no combination of of1000asacompromisethatkeptcomputationaldemandsfeasible.
3
neuralmeasureswillachieve100%accuracy.Toexploretheeffect DIChasbeencriticizedbecauseitcanselectmodelsthataretoocomplex.
of classification accuracy on recovery of the (true) dual-state Gelman et al. (2014) favor instead an information criterion that approximates
Bayesianleave-one-outcrossvalidation,WAIC(Watanabe,2013);foranumberof
| model, we | manipulated |     | state-level | misspecification |     | in  | terms of |     |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ----------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
checksweperformedonourextensivesimulationstudyDICandWAICproduced
theprobabilityofcorrectlyassigningatrialtoitstruegenerating
|                      |     |     |               |     |     |      |            | almost identical | results. | The | code we | provide | to apply | our | analyses | allows |
| -------------------- | --- | --- | ------------- | --- | --- | ---- | ---------- | ---------------- | -------- | --- | ------- | ------- | -------- | --- | -------- | ------ |
| state,whichwedenotep |     |     | .Forexample,p |     |     | =0 . | 8indicates |                  |          |     |         |         |          |     |          |        |
correct correct calculationofbothinformationcriteria,souserscanusetheirpreferredchoice.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 147
which is computed using samples from the joint posterior – which is increasingly greater than the variability of either of
parameter distribution. DIC is defined as DIC = D (θ) + 2p , the mixture components as the two means increasingly differ
D
whereD (θ) isthedevianceatthemeanofthesampledposterior – there is an increase in the standard deviation of the trial-
|     | θ   |     |     |     |     |     |     |     |     |     | η   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parameter vector , and p D is the effective number of model to-trial variability in drift rate ( ) estimate for the single-state
|     |     | =D−D | (θ) |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parameters,wherep D ,andDisthemeanofthesampled model. For the difference in drift rates that we investigated this
posteriorparameterdeviancevalues.LowervaluesofDICindicate increasewasonlymarginal,andtheslightlymorevariablesingle
thebettermodelforthedata(i.e.,themostparsimonioustradeoff driftratedistributionapproximatedthemixturedistributionquite
betweengoodnessoffitandmodelcomplexity). well (see also discussion around formulae (4) and (5)). This
We converted estimated DICs for each comparison of the approximationwilllikelybreakdownasthedifferenceinmeans
single- and dual-state models to model weights (for overview, becomes extreme, but as the difference we examined was quite
see Wagenmakers & Farrell, 2004). If the set of models under substantialitseemsunlikelythatvisualexaminationofgoodness-
considerationcontainthetruedata-generatingmodel,thenthese of-fitalonewouldbesufficientinpracticetodetectamisspecified
single-statemodel.
| weights provide | estimates |     | of the | posterior | probability | of each |       |      |        |          |     |          |            |        |
| --------------- | --------- | --- | ------ | --------- | ----------- | ------- | ----- | ---- | ------ | -------- | --- | -------- | ---------- | ------ |
|                 |           |     |        |           |             |         | Since | both | models | provided | a   | visually | compelling | fit to |
model(i.e.,theprobabilityconditionalonthedataofeachmodel
beingthetruemodelrelativetothesetofcandidatemodelsunder behavioraldata,wediscriminatedbetweenthesingle-anddual-
comparison).Otherwise,modelweightsprovideagradedmeasure statemodelsonthebasisofmodelweights,asisstandardinmost
ofevidenceratherthantheall-or-nonedecisionrulethatcanarise researchcomparingcompetingcognitivemodels.Theleftpanelof
when interpreting ‘raw’ information criteria. Model weights are Fig.3summarizesthemodelrecoverysimulation.Theweightin
also on the same scale for different data-set sizes (i.e., they fall favor of the dual-state model – the true data-generating model
on the unit interval), which allowed for simple comparison of – is shown on the y-axis. Light through to dark lines indicate
modelrecoveryacrossthesamplesizesthatweresystematically theamountofstate-levelmisspecification,whereclassificationto
manipulatedinourstudy. the true latent state was manipulated from chance performance
|                                                          |     |     |     |     |     |     | (p      | =   | 0 . 5, lightest |     | line) through | to  | perfect classification |     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------- | --- | --------------- | --- | ------------- | --- | ---------------------- | --- |
| Modelweightsarecalculatedbyfirstconsideringdifferencesin |     |     |     |     |     |     | correct |     |                 |     |               |     |                        |     |
DICforeachmodelfittoagivendataset:∆ ( )=DIC −minDIC, (p = 1, darkest line). The key comparison is the ability to
|     |     |     |     | i DIC |     | i   | correct |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
where minDIC is the lowest (i.e., best) DIC among the set of K identifythetruelatentgeneratingstateonthebasisofcognitive
modelsunderconsideration.Then,theDIC-basedweightformodel modelsfittobehavioraldata,acrossarangeofneurally-informed
w( )
i, DIC ,fromthesetofK modelsisgivenas classificationaccuracies.
i
Asexpected,evidenceinfavorofthedual-statemodelincreased
|     |  −1∆ |     |   |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
exp DIC as the number of trials in the synthetic data increased (larger
2 i
w( DIC )= . (1) values on the x-axis). This was, however, heavily influenced by
i
K    the amount of state-level misspecification. In our simulations,
|     | exp −1∆ | DIC |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2 k thisrepresentsthecapacityoftheneuraldatatoreliablyclassify
k=1
trialstotheirtruelatent(data-generating)state.Wheneverstate-
Wecalculatedmodelweightsforpairwisecomparisonsbetween levelmisspecificationwasabovechance(i.e.,p > 05),the .
correct
thesingle-anddual-statemodels.Allsyntheticdataweregener- evidenceinfavorofthedual-statemodelincreasedwithincreasing
atedfromthedual-statemodelsoourprimaryoutcomemeasure
|                                                |          |     |                |       |        |            | sample   | size.   | In particular,                        |             | it reached       | ceiling | by a sample    | size |
| ---------------------------------------------- | -------- | --- | -------------- | ----- | ------ | ---------- | -------- | ------- | ------------------------------------- | ----------- | ---------------- | ------- | -------------- | ---- |
| was the weight                                 | in favor | of  | the dual-state | model | (i.e., | successful |          |         |                                       |             |                  |         |                |      |
|                                                |          |     |                |       |        |            | of 1000  | trials  | when                                  | state-level | misspecification |         | was completely |      |
| modelrecovery),givenbyasimplifiedformofEq.(1), |          |     |                |       |        |            |          |         | =                                     |             |                  |         |                |      |
|                                                |          |     |                |       |        |            | absent(p | correct | 1),andbytheupperlimitofthesamplesizes |             |                  |         |                |      |
   w e e x p l o r e d ( 1 0 0 0 0 t r i a l s ) f o r m o d e r a t e c l a s s i f i c a t i o n a c c u r a c y
|     |     | e x | p − 1 ∆ | D IC |     |     |     | ≥   | .   |     |     |     |     |     |
| --- | --- | --- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dua l ( p 0 7 ) . F o r m o re p l a u s i b le s a m p l e s i z e s , h o w e v e r ,
| w ( DIC )= |     |     | 2   |     |     | . (2) | c o r r | e c t |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | ----- | ------- | ----- | --- | --- | --- | --- | --- | --- |
dual     r e c o v e r y o f t h e t r u e m o d e l w a s m o r e m o d e s t . E v e n w i t h n o s t a t e -
|     | −1  | ∆   |   | + − | 1∆  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
exp sin gl e D I C e x p dual DIC le v e l m i s s p e c i f i c a t i o n , t h e w e i g h t f o r t h e d u a l - s t a t e m od e l n e v e r
|     |     | 2   |     |     | 2   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
exceeded0.8forsamplesizeslessthan250trials.Wenotethata
Wecalculatedmodelweightsaccordingto(2)forallrelevantcom- modelweightof0.8correspondstoadifferenceofapproximately3
parisons,andthenaveragedoverthe200MonteCarloreplicates unitsontherawDICscale.Smalldifferencesininformationcriteria
withineachstate-levelmisspecification(0.5,0.55,...,0.95,1)by such as this are often considered as providing little more than
samplesize(100,250,500,1000,2000,5000,10000)cellofthe
weakevidence(e.g.,Burnham&Anderson,2004;Kass&Raftery,
design.
1995;Raftery,1995).Evenplacingoptimisticboundsonthelevel
|     |     |     |     |     |     |     | of classification |     | accuracy |     | that is possible | with | real neural | data |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | -------- | --- | ---------------- | ---- | ----------- | ---- |
= .
2.5. Resultsanddiscussion (e.g., p 0 9), the weight for the dual-state model only
correct
|     |     |     |     |     |     |     | exceeded | 0.8 | at a sample |     | size of approximately |     | 400 | trials, and |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | --------------------- | --- | --- | ----------- |
Thesingle-anddual-statemodelsprovidedanexcellentfittoall did not reach a decisive level of evidence until the sample size
syntheticdatasets.Fig.2(c)showsthefitofthesingle-statemodel exceeded1000trials.
to an exemplary synthetic data set. It is perhaps surprising, but On a more technical point, when state-level misspecification
alsoinstructive,thatthemisspecifiedsingle-statemodelprovided was at chance (p = 0 . 5), the single-state model ideally
correct
suchapreciseaccountofdatageneratedfromtwodiscretelatent oughttogarnerincreasingevidencewithincreasingsamplesize
| states that | had different | data-generating |     | parameters. |     | It appears |     |     |     |     |     |     |     |     |
| ----------- | ------------- | --------------- | --- | ----------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
(i.e.,agradualshifttowardlowervaluesonthey-axis).Thisshould
thatthesingle-statemodelisabletomimicthedual-statemodel, occursincetheclassificationtodiscretestatesinthefittedmodel
at least for the parameter settings we investigated. Specifically, was completely uninformed by the true data-generating values,
when the drift rate is the only parameter that varies across sotheestimateddriftratesfortrialsclassifiedtotheon-andoff-
|     |     | v   | v   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
discrete states – where and , respectively, represent drift task states were close to identical. Under these conditions, the
|     |     | on  | off |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
rates for the on-task and off-task states, and p represents the dual-state model provides no predictive benefit over the single-
on
proportion of on-task trials – the estimated (single) drift rate state model, so we should favor the simpler single-state model,
of the misspecified single-state model approximates a weighted andincreasinglysoforlargersamplesizes.ExaminationofFig.3,
combination of the two: v × p + v × ( 1 − p ) . To however, indicates that this did not occur; model weight was
|     |     |     | on  | on off |     | on  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mimic the variability of the mixture of drift rate distributions independent of sample size. This result is due to a property

148 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
Fig.3. Modelrecoveryformediumeffectsizes.Theleftpanelshowstheweightinfavorofthedual-statemodeloverthesingle-statemodelinthemodelrecoverysimulations
ofthediscretestaterepresentation.They-axisrepresentstheDIC-derivedposteriormodelprobabilityofthedual-statemodel,thex-axisrepresentsthenumberoftrials
inthesyntheticdataset,andcolorgradationsrepresenttherangeinpcorrectofthestate-levelmisspecificationofthedual-statemodel.Therightpanelshowstheweightin
favorofthecovariatemodeloverthestandardmodelinthemodelrecoverysimulationsofthecontinuousdimensionrepresentation.They-axisrepresentstheDIC-derived
posteriormodelprobabilityofthecovariatemodelandcolorgradationsrepresenttherangeinR2ofthecovariatemeasurementprecisionofthecovariatemodel.Horizontal
graylinesindicatethepointofequivalentevidencebetweenthetwomodels(solidlines),andadifferenceofapproximately3DICunitsinfavorofthedual-statemodel(left)
andcovariatemodel(right;upperdashedlines)orthesingle-statemodel(left)andstandardmodel(right;lowerdashedlines).(Forinterpretationofthereferencestocolor
inthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
of the model selection criteria used here. DIC penalizes model behavioraldataisdifficultbutnotimpractical.Wenowinvestigate
complexitywithafixedoffset(theeffectivenumberofparameters, ageneralizationofthediscretestaterepresentationthatconsiders
p ), which means that the penalty against the dual-state model the latent state as a continuous dimension. In the context of
| D   |     |     |     |     |     | .   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
over the single-state model when p = 0 5 is (almost) a mindwandering,suchacontinuumcouldrepresentadynamically
correct
fixedvalueasafunctionofthesamplesizemanipulationinour fluctuating state where people drift into phases of more on-
study,hencetheapproximatelyflatlineaty = 0 . 4.Thisproblem task or more off-task focus, without imposing a rigid boundary
| would be    | addressed  | through | the         | use of     | model    | selection      | indices  |            |            |                |                |     |           |            |         |
| ----------- | ---------- | ------- | ----------- | ---------- | -------- | -------------- | -------- | ---------- | ---------- | -------------- | -------------- | --- | --------- | ---------- | ------- |
|             |            |         |             |            |          |                |          | between    | mutually   | exclusive      | states.        | The | idea      | underlying | the     |
| that are    | consistent | in the  | sense       | that they  | converge | to             | the true |            |            |                |                |     |           |            |         |
|             |            |         |             |            |          |                |          | continuous | dimension  | representation |                | is  | more      | general,   | though, |
| answer with | increasing |         | sample      | size, such | as       | Bayes factors. | At       |            |            |                |                |     |           |            |         |
|             |            |         |             |            |          |                |          | mirroring  | constructs | in             | many cognitive |     | theories, | such       | as the  |
| the time    | of this    | work,   | calculation | of Bayes   | factors  | for            | complex  |            |            |                |                |     |           |            |         |
gradedmemorabilityofdifferentitemsinarecognitionmemory
cognitivemodelssuchasthediffusionmodeliscomputationally
|     |     |     |     |     |     |     |     | experiment. | Indeed, | it was | to  | account | for just | such | graded |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ------ | --- | ------- | -------- | ---- | ------ |
extremelyexpensive.Thisisanactivefieldofresearchandwith
variabilitythatRatcliff(1978)introducedtrial-to-trialvariability
futuredevelopmentswehopetoincorporatesuchmodelselection indriftratesintothediffusionmodel,whichhassincebecomea
measures in our work (for a recent example, see Steingroever, standardassumption(i.e., η> 0).
Wetzels,&Wagenmakers,2016). The continuous dimension representation can be interpreted
| In summary, |     | our simulation |     | study indicates |     | that it | can be |              |     |               |      |       |       |          |        |
| ----------- | --- | -------------- | --- | --------------- | --- | ------- | ------ | ------------ | --- | ------------- | ---- | ----- | ----- | -------- | ------ |
|             |     |                |     |                 |     |         |        | in two ways. | The | first assumes | that | there | is an | external | stream |
difficulttoidentifydiscretelatentstatesonthebasisofcognitive
|     |     |     |     |     |     |     |     | of information, |     | which we | assume | throughout |     | to be some | form |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | -------- | ------ | ---------- | --- | ---------- | ---- |
modelsfittobehavioraldata.Ofcourse,itispossiblethatchanges
|                   |     |        |            |     |       |                |     | of neural | data, | that reliably | indexes | a   | latent | state, | such as |
| ----------------- | --- | ------ | ---------- | --- | ----- | -------------- | --- | --------- | ----- | ------------- | ------- | --- | ------ | ------ | ------- |
| to the parameters |     | of the | simulation | may | alter | these results. | For |           |       |               |         |     |        |        |         |
mindwandering.Inthemindwanderingliterature,forexample,
| example, | we could | manipulate |     | the ratio | of on-task | to  | off-task |     |     |     |     |     |     |     |     |
| -------- | -------- | ---------- | --- | --------- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
measuresofconnectivityandactivityofthedefaultmodenetwork
| trials in    | synthetic | data,      | the number       | of      | model  | parameters     | that |               |             |               |     |         |           |           |         |
| ------------ | --------- | ---------- | ---------------- | ------- | ------ | -------------- | ---- | ------------- | ----------- | ------------- | --- | ------- | --------- | --------- | ------- |
|              |           |            |                  |         |        |                |      | are increased |             | during phases | of  | reduced | attention | toward    | the     |
| differed     | across    | the latent | states           | and the | degree | of difference, |      |               |             |               |     |         |           |           |         |
|              |           |            |                  |         |        |                |      | primary       | task (e.g., | Andrews-Hanna |     | et al., | 2010;     | Christoff | et al., |
| or the level | of        | parameter  | misspecification |         | in     | the models     | fit  |               |             |               |     |         |           |           |         |
2009;Masonetal.,2007;Mittneretal.,2014;formeta-analysis,
| to the synthetic |     | data. | On the | basis of | the available | evidence, |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ----- | ------ | -------- | ------------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
seeFox,Spreng,Ellamil,Andrews-Hanna,&Christoff,2015).Inthis
however,weconcludethatobtainingcompellingevidenceforthe
identificationofmutuallyexclusivelatentstates–suchasphases case, moment-to-moment fluctuations in activity of the default
ofon-taskandoff-taskperformance–requiresverylargesample mode network could be considered an online index of mind
sizes(5000+trials)withmoderate(orbetter)neuralclassifiers, wandering. This stream of neural data can then be used as a
or moderate (or better) sample sizes with very good neural covariateinthecognitivemodel;specifically,single-trialmeasures
classifiers. Our intuition is that neither of these situations arise ofdefaultmodenetworkactivitycanberegressedontostructured
trial-by-trialvariationintheparametersofthemodel.Thisallows
inthemajorityofrealpsychologicalorneuroscienceexperiments.
|     |     |     |     |     |     |     |     | exploration | of  | the effect | of the | neural | covariate | on different |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ---------- | ------ | ------ | --------- | ------------ | --- |
Nevertheless,foralmostallsamplesizesweobtainedatleastsome
modelparametersandpermitsquantitativetestsofthecovariate-
| evidence     | in favor | of the | true  | model for | plausible | sample    | sizes |           |          |      |             |      |        |            |       |
| ------------ | -------- | ------ | ----- | --------- | --------- | --------- | ----- | --------- | -------- | ---- | ----------- | ---- | ------ | ---------- | ----- |
|              |          |        |       |           |           |           |       | parameter | pairings | that | provide the | best | fit to | behavioral | data. |
| (e.g., a few | hundred  | to     | a few | thousand  | trials)   | when data | were  |           |          |      |             |      |        |            |       |
partitionedtodiscretestatesonthebasisofneuralclassifiersthat This approach has the potential to provide insights regarding
performedwithinanimpressivebutplausiblerangeforrealdata howthelatentstate(e.g.,mindwanderingasindexedbyactivity
=0 . 7−0 .
(e.g.,p 85). of the default mode network) affects cognition (e.g., processing
correct
efficiency;driftrate)andconsequenttaskperformance(e.g.,more
3. Continuousdimensionrepresentation errors,slowerresponsetimes).
|     |     |     |     |     |     |     |     | The second |     | way to interpret | a   | continuous | dimension |     | is that |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---------------- | --- | ---------- | --------- | --- | ------- |
The first model recovery analysis indicated that identifying the neural measure provides a direct ‘readout’ of a process
discrete latent states on the basis of cognitive models fit to assumedinthecognitivemodel.Thisapproachallowsforprecise

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 149
tests of ‘linking propositions’ (Schall, 2004); explicit hypotheses driftrate.Specifically,driftratesweredistributedaccordingtothe
about the nature of the mapping from particular neural states valueofthenormalizedcovariate(d)andaregressioncoefficient
to particular cognitive states. As an example of this approach, ( β ),suchthatthedriftrate( v )ontrialiis:
Cavanaghetal.(2011)proposedthatresponsecautioninconflict
v ∼v+β·d +N ( 0 , η). (3)
tasks is modulated by connectivity between the subthalamic i i
nucleusandmedialprefrontalcortex.Totestthishypothesis,the Thecovariatemodelthusassumedthatthedriftrateontriali, v ,
i
authorsfirstestimatedsingle-trialmeasuresofthetabandpower hadameancomponentdefinedasalinearfunctionofanintercept,
from neural oscillations in ongoing EEG activity over the medial v ,representingaverageperformanceintheexperiment,andthe
prefrontalcortex,whichwasthenregressedontothevalueofthe magnitudeandvalenceoftheneuralmeasureontriali,d,scaled
i
decisionboundaryparameterofthediffusionmodel.Thissingle- byaregressioncoefficient, β ,whichisanindexofeffectsize,and
trial regressor approach estimates regression coefficients that a random component involving samples from a Gaussian distri-
indicate the valence and magnitude of the relationship between butionwithmean0andstandarddeviation η .Thismodelreflects
theneuralmeasureandobservedperformance,viathearchitecture theplausibleassumptionthatourmeasuredneuralcovariatehasa
ofthecognitivemodel.Cavanaghetal.(2011)foundthatincreased generativeinfluenceondriftrate(throughparameter β ),butthere
thetapowerledtoasubsequentincreaseinthedecisionboundary arealsounmeasured,randomlydistributedinfluencesondriftrate
(i.e., a positive value of the regression coefficient) for trials (throughparameter η ).
with high but not low conflict. A control analysis indicated that
thetapowerhadnotrial-levelrelationshipwithdriftrate(i.e.,a
3.1.2. Effectsizeoftheneuralcovariate
regression coefficient centered at zero), indicating a selective β
We matched the effect size ( ) studied in the continuous
effectoftheneuralmeasureonamodelparameter.Thisexample
dimensionrepresentationtotheeffectsizestudiedinthediscrete
highlightshowsingle-trialregressionpermitsquantitativetestsof
statesimulationsintermsoftheproportionofvarianceaccounted
hypothesesaboutbrain–behaviorrelationships.
for by the neural information. Specifically, if p represents the
Regressingneuraldataontotheparametersofcognitivemodels on
proportionofon-tasktrialsinthediscretestaterepresentation,and
atthesingle-triallevelhasthedesirablepropertythatitprovides
x andx ,respectively,representsampleddriftratesoftheon-task
atightquantitativelinkbetweenneuralandbehavioraldata(de a 1 ndoff- 2 taskstates,wherex ∼ N (v ,η ) andx ∼ N (v ,η ) ,
Hollander, Forstmann, & Brown, 2016). Furthermore, although 1 on on 2 off off
thentheweightedmeandriftrateofthemixtureis
we used custom scripts for all analyses reported here – because
we needed to automate a large number of replications – there M =p ·v +( 1−p )·v , (4)
discrete on on on off
are excellent, freely available programs that implement single-
withvariance
trial regression for hierarchical and non-hierarchical Bayesian
parameterestimationforthediffusionmodel(HDDMtoolboxfor V = p ·η2 +( 1−p )·η2 +p ·(v −M )2
discrete on on on off on on discrete
Python;Wiecki,Sofer,&Frank,2013),whichremovesbarriersto
implementationofthesemethods.IntheAppendix,weoutlinethe +( 1−p on )·(v off −M discrete )2. (5)
steps involved in performing single-trial regression and provide Substituting the values used in the discrete state simulations
accompanyingRcodetoimplementthesesteps. (p = 0 . 8, v = 2, v = 1, and η = η = 1) into (4)
on on off on off
Inthissection,weassessedwhetherthetrial-by-trialinfluence and(5),wegetM = 1 . 8andV = 1 . 16.Theproportion
discrete discrete
ofanexternalstreamofinformation,suchasaneuralmeasure,is
ofvarianceaccountedforbytheneuraldatainthediscretestate
identifiableinmodelsfittobehavioraldata.Inprevioussimulation
simulationswastherefore
studies,Wieckietal.(2013)foundthatsingle-trialcovariatesare
1 1
well recovered in a hierarchical estimation setting for moderate R2 =1− =1− =0 . 138 ,
effectsizesandmoderatenumberoftrialsintheexperiment.We discrete V discrete 1 . 16
buildonWieckietal.’sfindingstoexplorehowoftenamodelthat 
incorporatesasingle-trialneuralcovariate–whichwasthetrue whichgivesthemediumeffectsizeofr discrete = R2 discrete =0 . 371.
model in all cases – was preferred over the ‘standard’ diffusion We used a comparable definition of effect size for the
modelthatusesnotrial-levelcovariates. continuous dimension representation. If the neural data is
distributedasd ∼ N ( 0 , V ) withregressioncoefficient β and
neural
3.1. Generatingsyntheticdata base drift rate variability x ∼ N ( 0 ,η) ,4 then it follows that the
covariatemodelin(3)hasvariance
Syntheticdataweregeneratedfromadiffusionmodelwhere
V
=η+β·V ,
a neural signal modulated individual-trial drift rates: trials with continuous neural
larger-than-average neural signals had larger-than-average drift withproportionvariance
rates and trials with smaller-than-average neural signals had
β·V
smaller-than-average drift rates. We assumed that the neural R2 = neural . (6)
covariate would be pre-processed and normalized prior to
continuous η+β·V
neural
modeling.Tothisend,wesimulatedasinglevalueoftheneural Rearranging(6)andsettingR2 =R2 =0 . 138,weget
covariate for every synthetic trial via random draws from the continuous discrete
standardnormaldistributionandexploredtheeffectoftheneural η·R2
covariateonrecoveryofthedata-generatingmodel. β = V ( 1− con R tin 2 uous ) =0 . 16 ,
neural continuous
which is the value of the regression coefficient we used to
3.1.1. Covariatemodel
generate synthetic data. This value is broadly representative of
Synthetic data were generated from a model that assumed
thefewpreviousstudiesthathavereportedsingle-trialregression
trial-to-trial variability in drift rate had systematic fluctuations,
viatheneuralcovariate,andunsystematic(random)fluctuations,
η
via parameter , which we refer to as the Covariate model. We
assumed that the trial-level neural covariate was mapped via 4 HerewesetVneural =1withoutlossofgeneralityandsimilarlybothmeansat
simple linear regression to structured trial-by-trial variation in zeroasweareonlyconcernedwithproportionsofvariance.

150 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
coefficientsinempiricalstudiesusingamodel-basedneuroscience Wereportcovariatemeasurementprecisionbelowasthecoef-
framework; β ≈ 0 . 20 for drift rate effects in Nunez et al. ficientofdetermination(R2)ratherthanPearsoncorrelationcoef-
| (2017),and | β ≈ | 0 . 09and0.04forresponsethresholdeffectsin |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ficient(r).Thisallowsfordirectinterpretationastheproportionof
Cavanagh et al. (2011) and Frank et al. (2015), respectively. All variancethatthenoise-perturbed,fittedvalueofthecovariateac-
other parameters of the covariate model were set to the same countsforinthetruedata-generatingvalueoftheneuralcovariate.
values as in the simulation of the on-task state of the discrete Theseresultsprovideabenchmarkfortheminimumlevelofmea-
representation. surementprecisionrequiredforidentifiabilityofcognitivemodels
Weagaingeneratedsyntheticdatasetsfromthesamerangeof thatincorporatesingle-trialcovariates.
samplesizesasinthepreviousanalysis;200syntheticdatasets
fromthecovariatemodelforeachofsamplesizes100,250,500,
3.3. Parameterestimationandmodelselection
1000,2000,5000,and10000trials.
|     |     |     |     |     |     |     |     | We estimated |     | model | parameters | using | identical | methods to |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | ---------- | ----- | --------- | ---------- |
3.2. Modelspecification
thosedescribedintheanalysisofthediscretestaterepresentation,
withtheonlyadditionthatwespecifiedapriordistributionforthe
Wefittwotypesofdiffusionmodelstoeachsyntheticdataset: ( , ,−3 , )
|     |     |     |     |     |     |     |     | covariateparameterofthecovariatemodel:N |     |     |     |     | 0 1 | 3 . |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | --- | --- |
thecovariatemodelanda‘standard’model.Thecovariatemodel
|            |               |     |           |          |       |                  |     | Model | selection | was | also conducted | in  | a parallel | manner to |
| ---------- | ------------- | --- | --------- | -------- | ----- | ---------------- | --- | ----- | --------- | --- | -------------- | --- | ---------- | --------- |
| was fit to | all synthetic |     | data sets | with the | drift | rate assumptions |     |       |           |     |                |     |            |           |
thefirstanalysis.Ourprimaryaimwastodeterminethecovariate
| specified | in (3). | The second | model | neglected |     | the information |     |     |     |     |     |     |     |     |
| --------- | ------- | ---------- | ----- | --------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
measurementprecisionrequiredtoobtainevidenceinfavorofthe
| contained | in the | neural | covariate | altogether, |     | instead | attributing |           |       |          |          |        |            |           |
| --------- | ------ | ------ | --------- | ----------- | --- | ------- | ----------- | --------- | ----- | -------- | -------- | ------ | ---------- | --------- |
|           |        |        |           |             |     |         |             | covariate | model | over the | standard | model. | Therefore, | we report |
trial-to-trialvariabilityindriftratetounsystematicsourcesviathe
η parameter;thatis, the model weight in favor of data generated from the covariate
(i.e.,true)modeloverthestandardmodel,following(2).
| v ∼N (v, | η). |     |     |     |     |     |     |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i
WerefertothissecondmodelastheStandardmodel,reflectingits
3.4. Resultsanddiscussion
dominantstatusintheliterature(Ratcliff,1978;Ratcliff&McKoon,
2008). All models provided an excellent fit to synthetic data so we
Whentheneuralsignalismeasuredwithperfectprecision,the againadjudicatedbetweenthemusingmodelweights.Theright
truelatentdata-generatingmodel–thecovariatemodel–should panelofFig.3summarizesmodelrecoveryinasimilarformatto
| be favored | over | the standard |     | model. | Such high | measurement |     |     |     |     |     |     |     |     |
| ---------- | ---- | ------------ | --- | ------ | --------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
theleftpanel.Largervaluesonthey-axisindicatemoreevidence
precision,however,isnotpossibleinrealneuraldata.Toexamine
|            |          |        |      |        |                |     |         | for the true | (covariate) |     | model | over the | standard | model. Line |
| ---------- | -------- | ------ | ---- | ------ | -------------- | --- | ------- | ------------ | ----------- | --- | ----- | -------- | -------- | ----------- |
| the effect | of noisy | neural | data | on the | identification | of  | a model |              |             |     |       |          |          |             |
darknessindicatesthelevelofcovariatemeasurementprecision,
| incorporating     | a             | neural     | covariate, | we          | manipulated | the     | level of   |                            |                                                |                 |           |                 |                      |               |
| ----------------- | ------------- | ---------- | ---------- | ----------- | ----------- | ------- | ---------- | -------------------------- | ---------------------------------------------- | --------------- | --------- | --------------- | -------------------- | ------------- |
|                   |               |            |            |             |             |         |            | where measurement          |                                                | precision       |           | was manipulated |                      | from complete |
| noise in          | the covariate | that       | was        | fit to the  | synthetic   | data.   | That is,   |                            |                                                |                 |           |                 |                      |               |
|                   |               |            |            |             |             |         |            | noise(R2                   | =0,lightestline)throughtoperfectmeasurement(R2 |                 |           |                 |                      | =             |
| we systematically |               | diminished | the        | correlation |             | between | the data-  |                            |                                                |                 |           |                 |                      |               |
|                   |               |            |            |             |             |         |            | 1, darkest                 | line).                                         | As before,      | the       | key comparison  | was                  | the capacity  |
| generating        | value         | of the     | covariate  | and         | the fitted  | value   | of the     |                            |                                                |                 |           |                 |                      |               |
|                   |               |            |            |             |             |         |            | to identify                | the                                            | true generating |           | model           | in neurally-informed |               |
| covariate,        | which         | we refer   | to as      | covariate   | measurement |         | precision. |                            |                                                |                 |           |                 |                      |               |
|                   |               |            |            |             |             |         |            | versus neurally-uninformed |                                                |                 | cognitive | models          | fit                  | to behavioral |
Thismanipulationmimicsthesetupofrealexperimentswherewe
data.
(aimto)obtainneuralmeasuresthatarenoise-perturbedproxies
|     |     |     |     |     |     |     |     | Evidence | in favor | of the | true | model generally |     | increased with |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------ | ---- | --------------- | --- | -------------- |
tothetrueneuralstate.
thenumberoftrialsinsyntheticdata.Asexpected,however,this
| To systematically |     | manipulate |     | covariate | measurement |     | preci- |     |     |     |     |     |     |     |
| ----------------- | --- | ---------- | --- | --------- | ----------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
wasinfluencedbythelevelofcovariatemeasurementprecision.
| sion, for | each synthetic |     | data set | we generated |     | a new | set of ran- |     |     |     |     |     |     |     |
| --------- | -------------- | --- | -------- | ------------ | --- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Whenthecovariatewasmeasuredwithverylowprecision–where
| dom variables | that       | served    | as the | neural | covariate | in the | models     |            |       |                  |     |           |           |           |
| ------------- | ---------- | --------- | ------ | ------ | --------- | ------ | ---------- | ---------- | ----- | ---------------- | --- | --------- | --------- | --------- |
|               |            |           |        |        |           |        |            | the fitted | value | of the covariate |     | explained | less than | 5% of the |
| that were     | fit to the | synthetic | data.  | The    | set of    | random | variables, |            |       |                  |     |           |           |           |
variationinthedata-generatingcovariate–samplesizehadalmost
whichwerefertoas‘fittedcovariates’,hadcorrelationswiththe
noinfluenceonrecoveryofthetruemodel.Thisimpliesthatwhen
| data-generating | value | of  | the covariate |     | ranging | from r | = 0 − 1 |     |     |     |     |     |     |     |
| --------------- | ----- | --- | ------------- | --- | ------- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- |
in increments of 0.1. The mean (zero), variance (one) and shape neuraldataarepoorlymeasured,orwhentheneuralmeasureis
(normal)ofthefittedcovariateswerethesameasthatoftheco- onlyaveryweakproxytothetruelatentprocess,thenabinary
variatedistribution.5 decisionwouldselectthestandardmodeloveraneurally-informed
model.Thatis,assumingunsystematicacross-trialvariationindrift
ratewouldbemoreparsimoniousthanregressinganoverlynoisy
neuralmeasureontodriftrate.
5
Underthismodelofmeasurementnoise,therelationshiptotheproportionof Perhaps surprisingly, evidence for the true model converged
varianceindriftratesexplainedbymindwanderingismoretransparentthanin
|     |     |     |     |     |     |     |     | very slowly | as a | function | of sample | size. Even | when | the neural |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | -------- | --------- | ---------- | ---- | ---------- |
thediscretecasewheremeasurementnoiseisintermsoftheproportionofcorrect
|                  |        |              |     |            |             |        |          | covariatewasperfectlymeasured(R2 |     |     |     | =1),theweightforthetrue |     |     |
| ---------------- | ------ | ------------ | --- | ---------- | ----------- | ------ | -------- | -------------------------------- | --- | --- | --- | ----------------------- | --- | --- |
| classifications. | To see | this, denote | the | proportion | of variance | in the | measured |                                  |     |     |     |                         |     |     |
covariate(MC)byw,andtherandomvariablesrepresentingthesystematiceffect
modeldidnotexceed0.8untilalmost2000trialswereobserved;
| ofthecovariateandmeasurementnoisebyD |     |     |     |     | ∼ N(0,1)andM |     | ∼ N(0,1), |                |     |        |          |              |       |            |
| ------------------------------------ | --- | --- | --- | --- | ------------ | --- | --------- | -------------- | --- | ------ | -------- | ------------ | ----- | ---------- |
|                                      |     |     |     |     |              |     |           | the comparable |     | sample | size for | the discrete | state | simulation |
respectively.Hence,MC=w·D+(1−w)·M,andsoMC∼N(0,1)asrequired.
was250trials.Formoreplausiblemeasurementprecision–say,
Consequently,theoveralldriftra√terandomvariablewiththemeasuredcovariateis
V ∼v+β·MC=v+β·D+N(0, 1+β·(1−w)).Theseresultsshowtheadditive approximately 33% – the weight for the true model exceeded
Gaussianassumptioncausesthedifferencebetweenmeasurementnoiseandthe 0.8 only when sample size exceeded approximately 4000 trials.
randomeffectsonthedriftrateunrelatedtothecovariatenottobeidentifiable,with This result, and similar comparisons across the panels of Fig. 3,
t h e c o m b in√at io n c o n st i tu ti n g w h a t m i gh t b e c al le d t h e ‘ ‘e f fe c t i v e ’’ l e ve l o f n o is e . suggeststhatthediscretestateapproachisamorepowerfuluse
| G i ve n , r = | w , o u r m | a n ip u l at | io n o f ri s | a m a n ip u | la ti o n o f | t h e e f f e c ti | v e n o is e le v e l | ,   |     |     |     |     |     |     |
| -------------- | ----------- | ------------- | ------------- | ------------ | ------------- | ------------------ | --------------------- | --- | --- | --- | --- | --- | --- | --- |
correspondingeithertoachangeinthelevelofmeasurementnoise,thelevelof ofneuraldatathanthesingle-trialcovariateapproach,atleastfor
unrelatedeffectsondriftrates,orsomecombination.Wemaintainthedistinction theparametersettingsandeffectsizeexploredhere.Thatis,neural
betweenthetwoconstituentsofeffectivenoiseinourdescriptionofresultsgiven
datamoreheavilyconstrainmodelrecoverywhenusedasabinary
itmakesclearthelinktothediscretecase,whereinbothcasestherangeofthe
indicatorofthelatentstatethanwhenregressedontotrial-by-trial
measurementnoisemanipulationisbetweennoeffectandthemaximaleffectsize
(i.e.,0.138=β/(1+β),whereβ=0.16). variationinmodelparameters.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 151
Fig.4. Modelrecoveryforlargeeffectsizes.Theleftpanelshowstheweightinfavorofthedual-statemodeloverthesingle-statemodelforthediscretestaterepresentation.
Therightpanelshowstheweightinfavorofthecovariatemodeloverthestandardmodelforthecontinuousdimensionrepresentation.Allotherdetailsareasdescribedin
Fig.3.
4. Recoveringneurally-informedcognitivemodelswhenneu- neuroscience literatures (e.g., Borst & Anderson, 2015; Mittner
raldatahavealargeeffectsize etal.,2014;Turner,Forstmannetal.,2013;Turner,VanMaanen,
& Forstmann, 2015). However, there have been few systematic
The foregoing analyses indicate that when equated on a studies of the benefits to model recovery that such an approach
medium effect size, neurally-informed discrete state models maybear.Wefoundthat,whentheneuraldatacandiscriminate
are more reliably recovered than neurally-informed continuous a moderate effect on performance, it can be difficult to reliably
dimensionmodels.Inthissection,weconfirmthatwhenendowed identifymutuallyexclusivelatentstateswhenneurally-informed
with a sufficiently large effect size the true model is well cognitive models are applied to behavioral data. As expected,
recovered in both the discrete state and continuous dimension modelrecoverywasverygoodwhenthesyntheticexperimental
representations. This result implies that both discrete and design was far removed from typical experiments (i.e., large
continuousrepresentationscanindeedbeidentifiedinbehavioral samplesize,goodneuralclassificationaccuracy).Modelrecovery,
data when the information contained in neural data relates to a however, was still within acceptable bounds even with more
sufficientlystrongeffect. feasibleexperimentaldesigns(i.e.,between500and1000trials)
We generated synthetic data sets where the neural data withmoderateclassificationaccuracy.
stronglyidentifiedthelatentstate.Specifically,forthecontinuous Incontrast,whenwerelaxedtheassumptionthatlatentstates
dimensionrepresentationwesetthevalueoftheneuralcovariate are discrete, we found that a latent state that can dynamically
to β = 0 . 5,withallotherparametersettingsasdescribedinthe movealongacontinuousdimensionsubstantiallyworsenedmodel
previous section. Following (6) this gives an effect size of r = recovery even though mind wandering accounted for the same
.
0577. An equivalent effect size can be obtained in the discrete proportion of variance (i.e., had the same effect size) in the
state representation in multiple ways. We chose to enhance the continuous and discrete versions. Model recovery was relatively
differencebetweentheon-andoff-taskstatesintermsofalarger poor for the sample sizes typically observed in psychological
driftratefortheon-taskstate( v on =2 . 414)andassuminganequal experiments(i.e.,upto1000trialsperparticipant),andconvincing
ratioofon-tasktooff-tasktrials(p on = 0 . 5),withnochangesin evidence for the true data-generating model was only obtained
otherdata-generatingparameters.Allotherdetailswereidentical withsamplesizesofapproximately5000trialsormore,evenwhen
to those used in the previous simulations, including the data neural covariates were (hypothetically) measured with perfect
generation, sample size, introduction of noise to the (synthetic) precision.Thisresultimpliesthatwhentheneuralcovariateisonly
neuraldata,modelspecification,parameterestimation,andmodel a distant proxy to the true data-generating process, a standard
selectionmethods. model that is ignorant with respect to neural data will often be
Fig.4showsrecoveryofthetruemodelwithalargeeffectsize preferredover aneurally-informedcognitive model,and, within
inthediscretestateandcontinuousdimensionrepresentations.A reason,thisisnotdependentonsamplesize.
strikingfindingwashowquicklytheevidenceforthetruemodels Webelievetheseresultshighlighttwoimportantissuesinthe
converged as a function of the noise in neural data (state-level useofneurally-informedcognitivemodels.Thefirst,moreobvious
misspecificationandcovariatemeasurementprecisionintheleft issueisthatwemustmaximizetheprecisioninourmeasurement
and right panels, respectively), even at relatively small samples of neural data. The second, more subtle issue is that we must
sizes(i.e.,250–500trials)andmoderatelevelsofnoise.Although usetheory-based,hypothesis-driventestsofneuralcovariateson
recovery of the continuous dimension representation was much modelparameters;thatis,wemustaimtomaximizethepossible
improvedforlargeversusmediumeffectsizes,thetruemodelin relationship between the fitted value of the covariate and the
thediscretestaterepresentationwasstillrecoveredmorereliably true data-generating process. The first issue can be addressed
whenequatingsamplesizeandnoiseinneuraldata. usingemergingtechnologiessuchasultra-highfieldMRI,which
allowsonetomeasurethebrainwithexcellentspatialresolution.
5. Conclusions Research using ultra-high field MRI also helps to address the
second issue as it requires a region of interest approach that is
We investigated whether informing cognitive models with necessarilyhypothesisdriven.
neuraldataimprovestheabilitytoidentifylatentcognitivestates. Our conservative conclusion is, therefore, that neural data
This approach is increasingly common in the psychology and aidsmodelidentificationundersomecircumstances.Inparticular,

152 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
model recovery improved when the latent state was assumed individual differences cannot be estimated, with hierarchical
to consist of discrete stages (vs. continuous dimension). The models often displaying ‘‘over-shrinkage’’ (i.e., estimating the
discrete approach had greater power in the sense that a given same parameter value for all participants). For simplicity, we
effect could be identified with smaller sample sizes, reflecting restricted our analyses to the simpler case of recovering latent
more efficient use of neural data. This finding may be due to cognitivestatesinindividualparticipants,whichremovesatleast
the parameter governing trial-to-trial variability in drift rate some sources of variability that are present in the hierarchical
η
( ) having a better capacity to compensate for variance arising case(e.g.,across-participantvariabilityintheproportionoftrials
fromtheneuralcovariateundertheassumptionofacontinuous fromeachoftwodiscretelatentstates).Weleavetheseinteresting
dimension than a discrete state representation. Nevertheless, in questionsaboutmodelrecoveryinhierarchicalsettingstofuture
| practice | this finding | is particularly |     | important | since | experiments |     | research. |     |     |     |     |     |     |     |
| -------- | ------------ | --------------- | --- | --------- | ----- | ----------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
thatrecordneuralmeasuressuchasfMRIorEEGactivityduring Finally, we note that the discrete and continuous approaches
task completion are often limited in the number of trials that need not involve neural data, although we considered such
can be collected. Reassuringly, when the neural data exerted a hypothetical scenarios here. A variable derived at the level of
| large effect | on behavior | (although |     | not such | a   | large effect | as to |               |         |     |                 |     |        |     |             |
| ------------ | ----------- | --------- | --- | -------- | --- | ------------ | ----- | ------------- | ------- | --- | --------------- | --- | ------ | --- | ----------- |
|              |             |           |     |          |     |              |       | single trials | – which | can | be incorporated |     | within | a   | discrete or |
beimplausibleatleastinsomecircumstances)boththediscrete continuousapproach–canbeextractedfromanypropertyofthe
stateandcontinuousdimensionrepresentationshadgoodmodel task environment that is relevant to performance. For example,
recovery. Even under this condition, however, relative to the Hawkins, Hayes, and Heit (2016) studied the similarity between
standardmodel,theassumptionofmutuallyexclusivelatentdata- studyandtestitemsinaninductivereasoningtask.Thesimilarity
generating states was more efficiently recovered than a latent relationsarespecifiedatthelevelofindividualitems,andthuscan
continuousdimension(cf.Fig.4).Finally,wenotethatefficiencyof beregressedagainstparametersofthecognitivemodelinthesame
modelrecoveryappearstobemoreheavilyinfluencedbytheeffect mannerasneuraldata.InHawkinsetal.’smodel,regressingsingle-
sizeratherthanparticularhyperparametersettings(cf.footnote1). trialitemsimilarityontothedriftrateparameterledtoapositive
| It is also | important | to  | note that | even | in the | large effect | size |     |     |     |     |     |     |     |     |
| ---------- | --------- | --- | --------- | ---- | ------ | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
regressioncoefficient,indicatingthatasitemsimilarityincreased
| case simple | visual | inspection | of model | fits | was | not sufficient | to  |            |                 |     |                 |     |          |          |     |
| ----------- | ------ | ---------- | -------- | ---- | --- | -------------- | --- | ---------- | --------------- | --- | --------------- | --- | -------- | -------- | --- |
|             |        |            |          |      |     |                |     | so too did | the probability |     | of generalizing |     | a target | property | to  |
rejectthestandardmodel;werequiredmodelselectionmethods. novelitemsaccordingtoaparticularfunctionalform.Thisexample
Fortunately, methods that are easily implemented based on illustratesthegeneralpointthatwiderincorporationofsingle-trial
standard Bayesian posterior sampling (e.g., DIC, WAIC) sufficed propertiesoftheexperiment–neuralorotherwise–incognitive
for detecting the presence of an effect of mind wandering in models has the potential to provide deeper insight into a broad
our simulations. However, more sophisticated model selection rangeofpsychologicalphenomena.
| methods | (e.g., Bayes | factors) | appear | to  | be required | to  | provide |     |     |     |     |     |     |     |     |
| ------- | ------------ | -------- | ------ | --- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
consistentevidence(i.e.,evidencethatbecomesstrongerassample
|                 |         |     |          |         |            |     |          | Appendix. | Implementingneurally-informedcognitivemodels |     |     |     |     |     |     |
| --------------- | ------- | --- | -------- | ------- | ---------- | --- | -------- | --------- | -------------------------------------------- | --- | --- | --- | --- | --- | --- |
| size increases) | against | the | presence | of mind | wandering. |     | That is, |           |                                              |     |     |     |     |     |     |
whenmindwanderingisnotpresent,atbestthemodelselection
InthisAppendix,weoutlinethestepsinvolvedinimplementing
methodsexploredherewillbeequivocalevenwithlargesamples.
|            |            |     |             |     |        |         |         | the discrete | state  | and  | continuous | dimension |     | representations |     |
| ---------- | ---------- | --- | ----------- | --- | ------ | ------- | ------- | ------------ | ------ | ---- | ---------- | --------- | --- | --------------- | --- |
| Regardless | of whether |     | one expects |     | neural | data to | exert a |              |        |      |            |           |     |                 |     |
|            |            |     |             |     |        |         |         | discussed    | in the | main | text. To   | accompany | the | examples,       | we  |
smallorlargeeffectonperformance,theassumptionofadiscrete provide code in the R programming language (R Core Team,
| or continuous | representation |     | will | likely | better | serve | different |            |           |           |     |          |         |           |     |
| ------------- | -------------- | --- | ---- | ------ | ------ | ----- | --------- | ---------- | --------- | --------- | --- | -------- | ------- | --------- | --- |
|               |                |     |      |        |        |       |           | 2016) that | is freely | available | on  | the Open | Science | Framework |     |
researchgoalsatdifferenttimes.Bothapproachesallowestimation
(osf.io/yt8q4).
ofeffectsize(cf.formulae(4)–(6)).Thecontinuousapproachalso
|     |     |     |     |     |     |     |     | This | outline provides |     | guidance | on  | the cognitive |     | modeling |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---------------- | --- | -------- | --- | ------------- | --- | -------- |
hastheattractivepropertythatameasureofeffectsizeisdirectly
componentofamodel-basedneuroscienceanalysisinrealdata.It
| estimated. | That is, | the output | of  | the | neural | covariate-model |     |     |     |     |     |     |     |     |     |
| ---------- | -------- | ---------- | --- | --- | ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
assumesthattheneuraldata–whetheritisfMRI,EEG,MEG,pupil
| parameter | relationship | –   | a regression | coefficient |     | – has | a simple |     |     |     |     |     |     |     |     |
| --------- | ------------ | --- | ------------ | ----------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
measurements,orothers–havebeenanalyzedinanappropriate
interpretation(assumingthattheneuralcovariateisnormalized):
|            |          |                |           |            |     |              |         | manner.   | It further | assumes  | that   | it is possible |     | to extract | at least     |
| ---------- | -------- | -------------- | --------- | ---------- | --- | ------------ | ------- | --------- | ---------- | -------- | ------ | -------------- | --- | ---------- | ------------ |
| the extent | to which | the            | estimated | regression |     | coefficient  | differs |           |            |          |        |                |     |            |              |
|            |          |                |           |            |     |              |         | one value | of the     | analyzed | neural | measure        | for | each       | trial of the |
| to zero    | provides | a standardized |           | measure    | of  | effect size. | Both    |           |            |          |        |                |     |            |              |
experiment.
approachesarealsorelativelyeasytoimplement.Thediscretestate
representationcanbeimplementedbysplittinganexperimental
|     |     |     |     |     |     |     |     | A.1. Implementingthediscretestaterepresentation |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
conditionintodiscretesetsoftrialsonthebasisofaneuralvariable
(e.g.,theoutputofaclassifier).Single-trialcovariatesarealready
|     |     |     |     |     |     |     |     | The discrete | state | representation |     | assumes |     | that the | observed |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | -------------- | --- | ------- | --- | -------- | -------- |
incorporatedasastandardfeatureofsomeestimationprograms
|     |     |     |     |     |     |     |     | data were | generated | by  | two or | more | discrete | latent | states. In |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- | ------ | ---- | -------- | ------ | ---------- |
(e.g.,HDDM,Wieckietal.,2013),removingapotentialbarrierto
implementation.IntheAppendix,wealsoprovidecustomRcode the main text, for example, we hypothesized that two latent
toimplementbothanalysisapproachesdiscussedinthispaper. states underlying task performance might correspond to an on-
Our analyses examined recovery of latent cognitive states task state, where attention is directed to external stimuli such
in individual (simulated) participants, though one could also asanexperimentaltask,andanoff-taskstate,whereattentionis
consider recovery of latent states across groups of participants. directed to internal stimuli such as mind wandering; these two
stateshavebeenproposedinthepopularperceptualdecoupling
| This could  | be investigated |           | with   | hierarchical     |              | Bayesian | models   |            |                  |           |      |            |              |           |             |
| ----------- | --------------- | --------- | ------ | ---------------- | ------------ | -------- | -------- | ---------- | ---------------- | --------- | ---- | ---------- | ------------ | --------- | ----------- |
|             |                 |           |        |                  |              |          |          | hypothesis | of mind          | wandering |      | (Smallwood | &            | Schooler, | 2015).      |
| that, among | other           | benefits, | allow  | for simultaneous |              | analysis | at       |            |                  |           |      |            |              |           |             |
|             |                 |           |        |                  |              |          |          | One could  | also hypothesize |           | more | than       | two discrete |           | states; for |
| the level   | of individuals  | and       | groups | (for             | an overview, |          | see Lee, |            |                  |           |      |            |              |           |             |
example,Cheyneetal.(2009)hypothesizedathree-statemodelof
2011).Suchanapproachallowsinformationtobepooledacross
engagement/disengagementfromtaskperformance.However,for
participantsinatheoreticallysensiblemanner,whichcanconfer
benefitstoparameterestimation,inparticularparameterstability. simplicity,werestrictedthemodelrecoveryanalysesinthemain
Furthermore, hierarchical Bayesian modeling can be applied to textandtheoutlineheretothemoreprominenttwo-statecase.
large samples of participants, where each participant may only The partition of individual trials into the latent states can be
complete a moderate number of trials. However, it is important derivedfromneuraldataintwomainways:usingsinglemeasures,
to note that if there are too few data for each participant then ormultiplemeasures.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 153
A.1.1. Identifyingdiscretelatentstatesfromasingleneuralmeasure ispreferredtoamodelfittodatathatisnot splitaccordingtoa
Step1:Thefirstmethodbeginswithidentificationofaneural neural signal. The model recovery properties of this comparison
signalrelatedtothelatentstatesofinterest.Inthemindwandering weretheprimaryfocusofthemaintext.
literature,forexample,activityofthedefaultmodenetwork(DMN)
tends to increase during phases of off-task focus and decrease
A.1.2. Identifyingdiscretelatentstatesfrommultipleneuralmeasures
during phases of on-task focus (Andrews-Hanna et al., 2010;
Thesecondmethoddifferstothefirstintermsofthenumber
Christoffetal.,2009;Masonetal.,2007;Mittneretal.,2014;for
ofneuralsignalsusedtoidentifythelatentcognitivestates,and
meta-analysis,seeFoxetal.,2015).Inthiscase,theneuralsignalof
thecomplexityofthemethodsusedtoinferthelatentstate.The
interestcouldbeasingle-trialmeasureofDMNactivity.Theneural
firstmethodassumedthattheneuralsignalcollapsedtoasingle
signalofinterestcanbesimpleinthesensethatitinvolvesasingle
value for each trial. The second method attempts to combine
measure(e.g.,stimulusevokedpupilresponse,P3ERPcomponent
multipleneuralsignalstoinferthelatentgeneratingstateoneach
overparietalcortex,orBOLDresponseindorsolateralprefrontal
trial. The general idea is that each neural signal might contain
cortex)or‘complex’inthesensethatitinvolvesanamalgamation
independentinformationaboutthelatentstatesosimplemethods
ofnumerousmeasures(e.g.,connectivitybetweenvariouscortical
ofaggregationmaylosediscriminatorypower.Amorepowerful
regions);thekeyrequirementisthatasinglevalueoftheneural
form of aggregation is through supervised learning algorithms,
signalcanbeextractedforeachtrial(methodscorrespondingto
thoughthisplacesanadditionalrequirementondatacollectionto
multipleneuralsignalsoneachtrialarediscussedinthefollowing
obtainthe‘labels’totrainaclassificationalgorithm.
subsection).Thespecificsforobtainingasinglevalueoneachtrial
might differ depending on the domain of study and the latent Step 1: The neural signals are extracted in a similar manner
statesofinterest;itcouldbethevalueoftheneuralsignalinaone toStep1ofAppendixA.1.1.However,hereweassumethereisa
secondintervalduringthepre-stimulusperiod,immediatelypost- setofneuralsignalsassociatedwiththelatentstatesofinterest;
stimuluspresentation,thefulltimecourseofatrial,orsomeother thestatesmightbeontaskandofftaskandthemeasuresmight
relevantinterval. be regional activity in the DMN and the task positive network,
Step2:Onceasinglevalueoftheneuralsignalisobtainedfor connectivitybetweentheDMNandthetaskpositivenetwork,and
eachtrial,theindividualtrialsaresortedinorderofthosewiththe stimulusevokedpupildiameter(cf.Mittneretal.,2014).
lowestvalueoftheneuralsignal(e.g.,lowDMNactivity)through Step 2: The general approach outlined here was performed
tothosewiththehighestvalueoftheneuralsignal(e.g.,highDMN in Mittner et al. (2014). This involves collecting neural signals
activity). Once sorted, the trials are split into separate groups. A as identified in Step 1 and behavioral data during regular task
simple option is to perform a median split of the DMN activity- performance that also involves occasional behavioral indicators
sortedtrialsontheassumptionthattrialswithlowerDMNactivity of the relevant latent states. In mind wandering research, for
aremorelikelytohavebeengeneratedbytheon-taskstateand example, participants are periodically asked to report whether
those with higher DMN activity are more likely to have been theirfocuswas‘moreontask’or‘moreofftask’inthepreceding
generatedbytheoff-taskstate.Amediansplitisacoarseapproach trial,thoughthisisnotaskedonalltrials.Thismethodtakesthese
and other methods can be used; for example, taking the lower self-reportratingsasanindicatorofthelatentstate–ontaskoroff
40% and upper 40% of trials, or using signs of bimodality in the task–andusesthemaslabelstotrainaclassificationalgorithm
distributionoftheneuralsignalasanindicatoroftheappropriate (e.g., a support-vector machine or neural network classifier) to
cut point for the sorted trials. The key requirement is that the ‘learn’ the distinct patterns of (the collection of) neural signals
neural signal is used to split individual trials into at least two that discriminate on-task from off-task self-report ratings. Most
discretegroupsoftrials. classificationalgorithmscontainadditionaltuningparametersthat
Step3:Oncethedatahavebeensplitbasedontheneuralsignal, shouldbeselectedinawaythatmaximizesapredictivescore.For
thecognitivemodelisfittothediscretegroupsoftrials.Critically, example,leave-one-outcross-validationinvolvestrainingseparate
thisfittingoccursasif thediscretegroupswerepartoftheexper- classifiersonallpossiblesetsofN−1trialsandtryingtopredictthe
imentaldesign.Inthemaintext,forexample,weassumedasin- labelofthe‘leftout’trial.Toavoidapossibleover-representation
gleexperimentalconditionwithnoexplicitmanipulation.When of one of the classes, which can distort the predictive accuracy
thedataweresplitaccordingtotheneuralsignal,weessentially measure (this can happen when, e.g., the criterion is the total
createdadatasetwithtwoconditionsthatcorrespondedtothe number of correct classifications), the area under the receiver-
twolatentstates;welabeledthese‘ontask’and‘offtask’.When operating characteristic curve criterion can be used since it
fitting the model to the latent states one can estimate partially balancesfalsealarmsandmisses.Aftertraining,theclassification
overlappingordistinctsetsofmodelparametersforthediscrete algorithmcanberefinedtofurtherimproveperformanceonthe
states.Thisisthesamelogicasfittingregularexperimentalma- selectedpredictivescore(e.g.,recursivefeatureelimination).Once
nipulations:whendifficultyismanipulatedacrossconditionsthe trainedandvalidated,thealgorithmprobabilisticallyclassifiesall
conventionalapproachistofreelyestimateadriftrateparameter unlabeled trials to the on-task or off-task state, based on the
foreachconditionwhileconstrainingothermodelparameterstoa correspondencebetweentheneuralsignalsoneachunlabeledtrial
commonvalue.Inthediscretestatescase,onemighthypothesize withtheneuralsignalonthelabeledtrials.
thatdriftratediffersacrossstatesbutotherparametersdonot.This Step3:Onceindividualtrialshavebeenclassifiedtotheon-task
correspondstotheassumptionthatthelatentstatesonlydifferin oroff-taskstates,thecognitivemodelisfittothediscretegroups
theefficiencyofstimulusinformationprocessing. oftrialsinthesamemannerasStep3ofAppendixA.1.1.Typical
The cognitive processes that might differ across latent states classificationalgorithmsusedforStep2producenotonlyalatent
oughttobedrivenbytheory.Ultimately,however,itcomesdown stateclassification,butalsoaprobabilityofcorrectassignmentto
toaquestionofmodelselection;doprocessesAandBdifferacross the state (i.e., p and p = 1 − p ). This uncertainty can be
on off on
latentstates,oronlyprocessA?Suchmodelcomparisonallowsone modeledinthelikelihoodfunctionforeachtrial’sdataasamixture
toaskthequestion:iftherearedifferencesincognitiveprocesses ofthelikelihoodsoftheon-taskandoff-taskstatestoaccountfor
betweenthelatentstates,whatisthemostlikelydifference?We noiseinclassificationaccuracy.Specifically,ifthedatafromtrial
arguethatthefinalandmostcriticalcomparisoniswhetherthe iareD andthelikelihoodofthesetofon-taskparametersgiven
i
simplestmodelofperformancedifferencesacrossthelatentstates classificationtotheon-taskstateforD isL (θ |D , on−task ) ,and
i on i

154 G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155
similarlyforofftask,then: trial( v ).Theaccompanyingcodeprovidesexplicitdetailshowto
i
|            |     |                 |     |     |         |             |     | com pu te t h | is s te p . |     |     |     |     |     |
| ---------- | --- | --------------- | --- | --- | ------- | ----------- | --- | ------------- | ----------- | --- | --- | --- | --- | --- |
| L (θ|D )=p | L   | (θ |D , on−task |     | )+p | L (θ |D | , off −task | ).  |               |             |     |     |     |     |     |
i on,i on i off,i off i In th e c o nt e x t ofsingle-trialregression,thereisanaddedin-
terpretationalbenefittousingaBayesianapproachtoparameter
β
A.2. Implementingthecontinuousdimensionrepresentation estimation:iftheposteriordistributionof doesnotcontainzero
thereislikelyasignificanteffectoftheneuralsignalonthemodel
The continuous dimension representation assumes that the parameter.Otherhypothesistestsarealsopossibleusingthepos-
observeddataweregeneratedbyaprocessthatdynamicallyvaries teriordistribution,forexample,estimationoftheSavage–Dickey
β
|     |     |     |     |     |     |     |     | Bayesfactor.Inferenceon |     | islessstraightforwardusingconven- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --------------------------------- | --- | --- | --- | --- |
alongacontinuouslatentdimension,relaxingtheassumptionthat
tionalparameterestimationmethodssuchasmaximumlikelihood
therearediscretelatentstates.Inthecontextofmindwandering,
estimation,thoughisstillpossible.
forexample,thisapproachassumesatrialwillfallatsomepoint
|         |           |      |       |                 |     |         |       | The extent | to which | the estimate | of  | the β parameter |     | differs |
| ------- | --------- | ---- | ----- | --------------- | --- | ------- | ----- | ---------- | -------- | ------------ | --- | --------------- | --- | ------- |
| along a | continuum | that | spans | from completely |     | on-task | focus |            |          |              |     |                 |     |         |
to0givesanestimateofthesignificanceoftheneuralsignalon
throughtocompletelyoff-taskfocus.Thepositionalongthislatent
themodelparameter,andhencecognitiveprocessofinterest.For
continuumdynamicallyvariesthroughoutthetask.
|     |     |     |     |     |     |     |     | example, if | we regressed | single-trial | measures |     | of (normalized) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------ | ------------ | -------- | --- | --------------- | --- |
Theaimofthismethodistoregressasingle-trialneuralsignal
|     |     |     |     |     |     |     |     | DMNactivityontodriftrateandobtainedanestimateof |     |     |     |     | β   | =−0 . 2, |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | -------- |
ontostructuredtrial-by-trialvariationinamodelparameter.Here
we outline and provide code to regress a single neural signal thisindicatesthatforeachunitincreaseinDMNactivitytherewas
onto a single model parameter. However, the methods can be adecreaseof0.2indriftrate.
easily extended to regress multiple neural signals onto a single Asinthediscretestateanalyses,thecognitiveprocessesthat
parameter (via multiple regression) or regress multiple neural might be dynamically modulated by a neural signal ought to be
drivenbytheory.However,again,thiscomesdowntoaquestion
| signals onto | multiple | model | parameters |     | (via separate |     | simple or |     |     |     |     |     |     |     |
| ------------ | -------- | ----- | ---------- | --- | ------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ofmodelselection;doestheneuralsignalhaveastrongersingle-
multipleregressionsfordifferentmodelparameters).
trialeffectonprocessAorBofthemodel?Asbefore,wearguethat
| Step | 1: The | neural signal | is  | extracted | in an | identical | manner |     |     |     |     |     |     |     |
| ---- | ------ | ------------- | --- | --------- | ----- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
themostimportantcomparisoniswhetherthemostparsimonious
| to Step | 1 of | Appendix | A.1.1. | For the | analyses | described | in  |     |     |     |     |     |     |     |
| ------- | ---- | -------- | ------ | ------- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
single-trialregressionmodelispreferredtoamodelfittodatathat
| the main | text | and outlined | here, | we  | assume | that the | neural |                                                          |     |     |     |     |     |     |
| -------- | ---- | ------------ | ----- | --- | ------ | -------- | ------ | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|          |      |              |       |     |        |          |        | isnot informedbyaneuralsignal.Themodelrecoveryproperties |     |     |     |     |     |     |
signalisnormalizedtoaGaussiandistributionwithmean0and
ofthiscomparisonweretheprimaryfocusofthemaintext.
| standard | deviation | 1. This | permits | examination |     | of simple | linear |     |     |     |     |     |     |     |
| -------- | --------- | ------- | ------- | ----------- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
relationshipsforthemappingbetweentheneuralsignalandthe
| model parameter. |     | Other | forms | of regression | that | do not | assume | References |     |     |     |     |     |     |
| ---------------- | --- | ----- | ----- | ------------- | ---- | ------ | ------ | ---------- | --- | --- | --- | --- | --- | --- |
simplelinearmappingsarepossiblebutwedonotexplorethose
here. Anderson,J.R.,Bothell,D.,Byrne,M.D.,Douglass,S.,Lebiere,C.,&Qin,Y.(2004).An
integratedtheoryofthemind.PsychologicalReview,111,1036–1060.
| Step | 2: The | hypothesized |     | neural | signal-model |     | parameter |     |     |     |     |     |     |     |
| ---- | ------ | ------------ | --- | ------ | ------------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
Andrews-Hanna,J.R.,Reidler,J.S.,Sepulcre,J.,Poulin,R.,&Buckner,R.L.(2010).
mappingisformulatedviasimplelinearregression.Forexample,in Functional–anatomicfractionationofthebrain’sdefaultnetwork.Neuron,65,
550–562.
themaintextweexploredacovariatemodelthatmappedasingle- Bastian,M.,&Sackur,J.(2013).Mindwanderingatthefingertips:Automaticparsing
trialneuralsignaltosingle-trialdriftrates(formula(3)ofthemain ofsubjectivestatesbasedonresponsetimevariability.FrontiersinPsychology,
text).Denotethenormalizedneuralsignald,regressioncoefficient 4,1–11.http://dx.doi.org/10.3389/fpsyg.2013.00573.
β v Borst,J.P.,&Anderson,J.R.(2015).Thediscoveryofprocessingstages:Analyzing
| ,anddriftrate |     | ,thenthesimplestcovariatemodelfordriftrate |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
EEGdatawithhiddensemi-Markovmodels.NeuroImage,108,60–73.
ontrialiis Brooks,S.P.,&Gelman,A.(1998).Generalmethodsformonitoringconvergence
ofiterativesimulations.JournalofComputationalandGraphicalStatistics,7,
| v ∼v+β·d |     |     |     |     |     |     | (A.1) | 434–455. |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- |
| i        | i   |     |     |     |     |     |       |          |     |     |     |     |     |     |
Brown,S.D.,&Heathcote,A.(2008).Thesimplestcompletemodelofchoicereaction
(we also assumed across-trial variability in drift rate in the time:Linearballisticaccumulation.CognitivePsychology,57,153–178.
covariatemodeldescribedinformula(3)ofthemaintext,which Burnham,K.P.,&Anderson,D.R.(2004).Multimodelinference:UnderstandingAIC
is omitted here for simplicity). This mapping assumes that the andBICinmodelselection.SociologicalMethods&Research,33,261–304.
Cavanagh,J.F.,Wiecki,T.V.,Cohen,M.X.,Figueroa,C.M.,Samanta,J.,Sherman,S.J.,
|     |     | v   |     |     |     |     | v   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
driftrateontriali, i ,hasameancomponent–theintercept, , etal.(2011).Subthalamicnucleusstimulationreversesmediofrontalinfluence
overdecisionthreshold.NatureNeuroscience,14,1462–1467.
representingaverageperformanceinthecondition/experiment–
|     |     |     |     |     |     |     |     | Cheyne, J. A., | Solman, G. J., Carriere, | J. S., | & Smilek, | D. (2009). | Anatomy | of an |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ------------------------ | ------ | --------- | ---------- | ------- | ----- |
thatismodulatedonatrial-by-trialbasisbythemagnitudeand error: A bidirectional state model of task engagement/disengagement and
valence of the neural signal on trial i, d i , scaled by a regression attention–relatederrors.Cognition,111,98–113.
β
coefficient, ,whichisanindexofeffectsize. Christoff, K., Gordon, A. M., Smallwood, J., Smith, R., & Schooler, J. W. (2009).
|            |     |            |               |     |     |               |     | Experience | sampling during | fMRI reveals | default | network | and | executive |
| ---------- | --- | ---------- | ------------- | --- | --- | ------------- | --- | ---------- | --------------- | ------------ | ------- | ------- | --- | --------- |
| The neural |     | signal can | theoretically | map | to  | any parameter | of  |            |                 |              |         |         |     |           |
systemcontributionstomindwandering.ProceedingsoftheNationalAcademy
the cognitive model of interest. When modulating single-trial ofSciencesoftheUnitedStatesofAmerica,106,8719–8724.
parameter values it is important to ensure that the regression Craigmile,P.F.,Peruggia,M.,&VanZandt,T.(2010).HierarchicalBayesmodelsfor
responsetimedata.Psychometrika,75,613–632.
(A.1)doesnotallowanysingle-trialparameterestimatestomove
deHollander,G.,Forstmann,B.U.,&Brown,S.D.(2016).Differentwaysoflinking
beyondfeasibleboundariesofthemodel(e.g.,asingle-trialvalue
|     |     |     |     |     |     |     |     | behavioral | and neural data | via computational |     | cognitive | models. | Biological |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------------- | ----------------- | --- | --------- | ------- | ---------- |
oftheresponsethresholdornon-decisiontimebelow0).Thiscan Psychiatry:CognitiveNeuroscienceandNeuroimaging,1,101–109.
beinstantiatedwitha‘check’intheparameterestimationroutine Forstmann,B.U.,Ratcliff,R.,&Wagenmakers,E.-J.(2016).Sequentialsampling
modelsincognitiveneuroscience:Advantages,applications,andextensions.
thatassignsverysmalllikelihoodtotrialswithinfeasiblesingle- AnnualReviewofPsychology,67,641–666.
trial parameter values, which results in low likelihood for the Fox,K.C.R.,Spreng,R.N.,Ellamil,M.,Andrews-Hanna,J.R.,&Christoff,K.(2015).
β The wandering brain: Meta–analysis of functional neuroimaging studies of
| correspondingestimateof |     |     | .Alternativelytheparametercanbe |     |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mind–wanderingandrelatedspontaneousthoughtprocesses.NeuroImage,111,
| transformedsothatitisunbounded. |     |     |     |     |     |     |     | 611–621. |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
Step 3: Once the single-trial regression is parameterized, Frank,M.J.,Gagne,C.,Nyhus,E.,Masters,S.,Wiecki,T.V.,Cavanagh,J.F.,&Badre,D.
(2015).fMRIandEEGpredictorsofdynamicdecisionparametersduringhuman
| the cognitive |     | model is | fit to | the behavioral |     | data. In | addition |     |     |     |     |     |     |     |
| ------------- | --- | -------- | ------ | -------------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- |
reinforcementlearning.TheJournalofNeuroscience,35,485–494.
to other model parameters, this involves estimating parameters Gelman,A.,Carlin,J.B.,Stern,H.S.,Dunson,D.B.,Aki,V.,&Rubin,D.B.(2014).Texts
corresponding to the mean component and the regression instatisticalscience,Bayesiandataanalysis(3rded.).CRCPress.
coefficientofthelinearregression( v and β in(A.1),respectively). Gomez,P.,Ratcliff,R.,&Perea,M.(2007).Amodelofthego/no–gotask.Journalof
ExperimentalPsychology:General,136,389–413.
| The neural | signal | (d ) is | provided | with the | data. | Together, | these |     |     |     |     |     |     |     |
| ---------- | ------ | ------- | -------- | -------- | ----- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- |
i Hawkins,G.E.,Hayes,B.K.,&Heit,E.(2016).Adynamicmodelofreasoningand
threecomponentsallowestimationofauniquedriftrateforeach memory.JournalofExperimentalPsychology:General,145,155–180.

G.E.Hawkinsetal./JournalofMathematicalPsychology76(2017)142–155 155
Hawkins, G. E., Mittner, M., Boekel, W., Heathcote, A., & Forstmann, B. U. Singmann, H., Brown, S., Gretton, M., & Heathcote, A. (2016). rtdists: Re-
(2015). Toward a model-based cognitive neuroscience of mind wandering. sponse time distributions. R package version 0.4-9. URL http://CRAN.R-
Neuroscience,310,290–305. project.org/package=rtdists.
Kass,R.E.,&Raftery,A.E.(1995).Bayesfactors.JournaloftheAmericanStatistical Smallwood,J.,&Schooler,J.W.(2006).Therestlessmind.PsychologicalBulletin,132,
Association,90,377–395. 946–958.
Killingsworth,M.A.,&Gilbert,D.T.(2010).Awanderingmindisanunhappymind. Smallwood,J.,&Schooler,J.W.(2015).Thescienceofmindwandering:Empirically
Science,330,932–932. navigating the stream of consciousness. Annual Review of Psychology, 66,
Lee,M.D.(2011).HowcognitivemodelingcanbenefitfromhierarchicalBayesian
487–518.
models.JournalofMathematicalPsychology,55,1–7. Smilek,D.,Carriere,J.S.,&Cheyne,J.A.(2010).Failuresofsustainedattention
Logan,G.D.,VanZandt,T.,Verbruggen,F.,&Wagenmakers,E.-J.(2014).Onthe
inlife,lab,andbrain:EcologicalvalidityoftheSART.Neuropsychologia,48,
abilitytoinhibitthoughtandaction:Generalandspecialtheoriesofanactof
2564–2570.
control.PsychologicalReview,121,66–95. Smith, P. L., & Ratcliff, R. (2004). The psychology and neurobiology of simple
Mason,M.F.,Norton,M.I.,VanHorn,J.D.,Wegner,D.M.,Grafton,S.T.,&Macrae,C. decisions.TrendsinNeurosciences,27,161–168.
N.(2007).Wanderingminds:Thedefaultnetworkandstimulus–independent Spiegelhalter,D.J.,Best,N.G.,Carlin,B.P.,&vanderLinde,A.(2002).Bayesian
thought.Science,315,393–395.
measuresofmodelcomplexityandfit.JournaloftheRoyalStatisticalSociety.
McVay,J.C.,&Kane,M.J.(2010).Doesmindwanderingreflectexecutivefunctionor
SeriesBStatisticalMethodology,64,583–639.
executivefailure?CommentonSmallwoodandSchooler(2006)andWatkins
Steingroever, H., Wetzels, R., & Wagenmakers, E.-J. (2016). Bayes factors for
(2008).PsychologicalBulletin,136,188–207.
reinforcement-learningmodelsoftheIowaGambingTask.Decision,3,115–131.
Mittner,M.,Boekel,W.,Tucker,A.M.,Turner,B.M.,Heathcote,A.,&Forstmann,
Teasdale,J.D.,Dritschel,B.H.,Taylor,M.J.,Proctor,L.,Lloyd,C.A.,Nimmo-Smith,
B.U.(2014).Whenthebraintakesabreak:Amodel–basedanalysisofmind
I.,&Baddeley,A.D.(1995).Stimulus-independentthoughtdependsoncentral
wandering.TheJournalofNeuroscience,34,16286–16295.
Nosofsky,R.M.,&Palmeri,T.J.(1997).Anexemplar–basedrandomwalkmodelof executiveresources.Memory&Cognition,23,551–559.
speededclassification.PsychologicalReview,104,266–300. Turner,B.M.,Forstmann,B.U.,Wagenmakers,E.-J.,Brown,S.D.,Sederberg,P.B.,&
Nunez,M.D.,Srinivasan,R.,&Vandekerckhove,J.(2015).Individualdifferencesin Steyvers,M.(2013).ABayesianframeworkforsimultaneouslymodelingneural
attentioninfluenceperceptualdecisionmaking.FrontiersinPsychology,8,1–13. andbehavioraldata.NeuroImage,72,193–206.
http://dx.doi.org/10.3389/fpsyg.2015.00018. Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013b). A
Nunez,M.D.,Vandekerckhove,J.,&Srinivasan,R.(2017).Howattentioninfluences methodforefficientlysamplingfromdistributionswithcorrelateddimensions.
perceptualdecisionmaking:Single-trialEEGcorrelatesofdrift–diffusionmodel PsychologicalMethods,18,368–384.
parameters.JournalofMathematicalPsychology,76,117–130. Turner,B.M.,VanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
RCoreTeam,2016.R:ALanguageandEnvironmentforStatisticalComputing. stractionsthroughneuroimaging:Theneuraldriftdiffusionmodel.Psychologi-
RFoundationforStatisticalComputing,Vienna,Austria.URLhttps://www.R- calReview,122,312–336.
project.org/. Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative
Raftery, A. E. (1995). Bayesian model selection in social research. Sociological representationofuncertainty.JournalofRiskandUncertainty,5,297–323.
Methodology,25,111–163. Vandekerckhove,J.,Tuerlinckx,F.,&Lee,M.D.(2008).ABayesianapproachto
Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85,59–108. diffusionmodelsofdecision–making.InV.M.Sloutsky,B.C.Love,&K.McRae
Ratcliff,R.,&McKoon,G.(2008).Thediffusiondecisionmodel:Theoryanddatafor
(Eds.),Proceedingsofthe30thannualconferenceofthecognitivesciencesociety
two–choicedecisiontasks.NeuralComputation,20,873–922.
(pp.1429–1434).CognitiveScienceSociety.
Ratcliff,R.,&Smith,P.L.(2004).Acomparisonofsequentialsamplingmodelsfor
Wagenmakers,E.-J.,&Farrell,S.(2004).AICmodelselectionusingAkaikeweights.
two–choicereactiontime.PsychologicalReview,111,333–367.
PsychonomicBulletin&Review,11,192–196.
Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion
Wagenmakers,E.J.,Farrell,S.,&Ratcliff,R.(2004).Estimationandinterpretationof
m
va
o
ri
d
a
e
b
l
i
:
li
A
ty
p
.
p
P
r
s
o
y
a
c
c
h
h
o
e
n
s
o
t
m
o
i
d
c
e
B
a
u
li
l
n
le
g
ti
w
n
i
&
th
R
c
e
o
v
n
ie
t
w
am
,9
i
,
n
4
a
3
n
8
t
–
re
4
a
8
c
1
t
.
iontimesandparameter 1/fαnoiseinhumancognition.PsychonomicBulletin&Review,11,579–615.
Robertson,I.H.,Manly,T.,Andrade,J.,Baddeley,B.T.,&Yiend,J.(1997).‘Oops!’: Watanabe,S.(2013).AwidelyapplicableBayesianinformationcriterion.Journalof
Performance correlates of everyday attentional failures in traumatic brain MachineLearningResearch,14,867–897.
injuredandnormalsubjects.Neuropsychologia,35,747–758. Weissman,D.H.,Roberts,K.C.,Visscher,K.M.,&Woldorff,M.G.(2006).Theneural
Schall,J.D.(2004).Onbuildingabridgebetweenbrainandbehavior.AnnualReview basesofmomentarylapsesofattention.NatureNeuroscience,9,971–978.
ofPsychology,55,23–50. Wiecki,T.V.,Sofer,I.,&Frank,M.J.(2013).HDDM:HierarchicalBayesianestimation
Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory: REM- ofthedrift-diffusionmodelinPython.FrontiersinNeuroinformatics,7,1–10.
retrievingeffectivelyfrommemory.PsychonomicBulletin&Review,4,145–166. http://dx.doi.org/10.3389/fninf.2013.00014.