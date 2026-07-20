JournalofMathematicalPsychology76(2017)104–116
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Variability in behavior that cognitive models do not explain can be
linked to neuroimaging data
∗
SebastianGluth ,JörgRieskamp
DepartmentofPsychology,UniversityofBasel,Switzerland
h i g h l i g h t s
• Manycognitivemodelsneglecttrial-by-trialvariabilityinbehavior.
• Thiscanbeovercomebyassumingvariabilityinmodelparameters.
• Asimplemethodtoeffectivelycapturethisvariabilityisintroduced.
• Thegoalistolinkvariabilityinmodelparameterswithneuroimagingdata.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Itisknownthatbehaviorissubstantiallyvariableevenacrossnearlyidenticalsituations.Manycognitive
Availableonline24May2016 models are not able to explain this intraindividual variability but focus on explaining interindividual
differencescapturedinmodelparameters.Insequentialsamplingmodelsofdecisionmaking,forinstance,
Keywords:
onesinglethresholdparametervalueisestimatedforeverypersontoquantifyhowmuchevidencemust
Cognitivemodeling
beaccumulatedforcommittingtoachoice.However,thisthresholdmayvaryacrosstrialsevenwithin
fMRI
subjectsandexperimentalconditions.Neuroimagingtoolssuchasfunctionalmagneticresonanceimaging
EEG
Intraindividualdifferences (fMRI)orelectroencephalography(EEG)canrevealmoment-to-momentfluctuationsintheneuralsystem
Bayes thatarelikelytocontributetofluctuationsinbehavior.Weproposethatneuralandbehavioralvariability
Decisionmaking couldbelinkedtoeachotherbyassumingandestimatingtrial-by-trialvariabilityinmodelparameters.To
illustrateourproposal,wefirsthighlightrecentstudiesinmodel-basedcognitiveneurosciencethathave
gonebeyondcorrelatingmodelpredictionswithneuroimagingdata.Thesestudiesmadeuseofvariance
inbehaviorthatremainedunexplainedbycognitivemodelingbutcouldbelinkedtospecificfMRIorEEG
signals.Second,wespecifyinatutorialanovelandefficientapproach,howtoextractsuchvarianceandto
applyittoneuroimagingdata.Ourproposalshowshowthevariabilityinbehaviorandtheneuralsystem
canprovideafruitfulsourceoftheorydevelopmentincognitiveneuroscience.
©2016ElsevierInc.Allrightsreserved.
1. Introduction cortex, to the parametric encoding of reward expectations and
rewardpredictionerrors(Bartra,McGuire,&Kable,2013;Clithero
Cognitive neuroscience and cognitive modeling pursue a &Rangel,2014;Garrison,Erdeniz,&Done,2013).Expectedvalues
common goal, which is to identify the various mechanisms that and prediction errors are particularly suitable for neuroimaging
give rise to observable behavior. It is therefore not surprising data analysis, because they vary on a trial-by-trial basis when
that their combination has become an extremely productive using reinforcement learning (RL) paradigms. The predictions of
research agenda. For example, model-based functional magnetic anRLmodel–thathasbeenfittedtothechoicedata–aresimply
resonanceimaging(fMRI;O’Doherty,Hampton,&Kim,2007)has regressedagainstthefMRIdata,revealingwhichbrainareastrack
been applied to link activation in the brain’s reward circuitry, thedevelopmentofexpectedvaluesorpredictionerrorsoverthe
including the ventral striatum and the ventromedial prefrontal courseoftheexperiment(e.g.,Daw,O’Doherty,Dayan,Seymour,
& Dolan, 2006; Gläscher, Hampton, & O’Doherty, 2009; Gluth,
Rieskamp,&Büchel,2014;O’Dohertyetal.,2004andPessiglione
∗ etal.,2008).
Correspondenceto:DepartmentofPsychology,UniversityofBasel,Mission-
sstrasse62a,4055Basel,Switzerland.Tel.:+41612670606. Importantly, a person will perceive an identical decision
E-mailaddress:sebastian.gluth@unibas.ch(S.Gluth). situation differently after receiving feedback, which leads to
http://dx.doi.org/10.1016/j.jmp.2016.04.012
0022-2496/©2016ElsevierInc.Allrightsreserved.

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 105
learning and consequently to different behavior across trials. ageneralformalizationofthisapproachrootedinBayesiantheory,
In contrast to learning situations, however, many psychological aswellasanaccompanyingexampletofacilitateunderstanding;
experiments use static environments, and potential changes in in Section 4 we discuss and exemplify parameter recovery
cognitiveprocessesfromtrialtotrialarenotaccountedfor.Evenan simulations as a precondition for the successful application of
RLmodeldoesnotpredictanytrial-by-trialvariability,aslongasno ourtechnique;inSection5wespecifyfurtherrequirementsand
learningtakesplace(e.g.,ifthefeedbackisfullypredictable).Inthe recommendations for models and neuroimaging data; Section 6
worstcase,somemodelsmakedeterministicpredictionsanddo provides a comparison with alternative ways to test for trial-
notallowaccountingforanyformofvariability.Traditionalmodels by-trial effects; finally, Section 7 concludes with a discussion of
ofdecisionmakingunderuncertainty,includingexpectedutility potential results that can be obtained with this technique and
theory (Von Neumann & Morgenstern, 1947), prospect theory theirimplicationsforfurtherdevelopmentsoftheoriesincognitive
(Kahneman & Tversky, 1979), but also some choice heuristics neuroscience.
(e.g., Brandstätter, Gigerenzer, & Hertwig, 2006), belong to this
category.Applyinganaïveerrortheoryallowsformulatingchoice 2. Model-basedneuroimagingstudiesofparametervariability
predictions in terms of a probability distribution (Hey, 1995;
Loomes & Sugden, 1995; McFadden, 2001; Rieskamp, 2008). Whereas reinforcement learning studies exploited the fact
Beyond that, sequential sampling models of decision making that many learning components such as reward expectations,
assumethatstochasticsamplingofevidenceunderliesthedecision prediction errors, but also learning rates and various forms of
process,whichallowsprobabilisticpredictionsofresponsetimes uncertainty are naturally changing (Behrens, Woolrich, Walton,
(Busemeyer&Townsend,1993;Smith&Ratcliff,2004).However, & Rushworth, 2007; Boll, Gamer, Gluth, Finsterbusch, & Büchel,
even probabilistic choice models make identical probabilistic 2013;Krugel,Biele,Mohr,Li,&Heekeren,2009;McGuire,Nassar,
predictionsforidenticalchoicesituations,sothatiftheprobability Gold, & Kable, 2014; Payzan-LeNestour, Dunne, Bossaerts, &
ofchoosingoptionAoveralternativeoptionBispredictedtobe, O’Doherty,2013),modelsofdecisionmaking(includingsequential
say,60%inonetrial,itwillalsobe60%whenthesamesituation samplingmodels)facetheproblemthatpredictionsoftenremain
isencounteredafewtrialslater.Thatthesubjectmightactually constantfromtrialtotrial.1 Infact,observationsareassumedto
chooseAinthefirstbutBinthesecondsituation(insteadofBand begeneratedbythesameprobabilitydistribution(e.g.,aWiener
thenA,orAinbothtrials,orBinbothtrials)hastobeattributedto diffusionprocess)andtobemutuallyindependent;thatis,theyare
unsystematicnoisethemodelcannotexplain. independent and identically distributed (i.i.d.) random variables
More precisely, when model parameters are assumed to be (Luce, 1986). Accordingly, early model-based fMRI studies on
fixed for specific conditions or persons, variability in behavior perceptualdecisionmakingrestrictedtheassociationofcognitive
isdifficulttoexplain(Lee&Wagenmakers,2014;Lewandowsky modelingandbraindatatocorrelationsacrosssubject(Forstmann
& Farrell, 2011). In sequential sampling models of decision etal.,2008).Inparticular,Forstmannandcolleaguesusedthelinear
making, for instance, a single decision threshold parameter ballisticaccumulator(LBA)model(Brown&Heathcote,2008)to
valueisestimatedthatreflectstheamountofevidencerequired modelchoicesandRTsinarandomdotmotiontaskwithdifferent
to make a choice (Bogacz, Brown, Moehlis, Holmes, & Cohen, conditions that prioritized speed, accuracy, or neither. The LBA
2006; Smith & Ratcliff, 2004). This parameter might vary across was estimated so as to allow specifying the level of response
differentconditions,forexample,wheneitherspeedoraccuracyis caution(i.e.,thedistancebetweenthedecisionthresholdandthe
emphasized.Butwithinasingleconditionthethresholdisassumed upperendofthestartingpointdistribution)inspeedandaccuracy
tobefixed,anditsvalueisusuallyestimatedbasedonhundredsor conditionsseparately.Asexpected,subjectswerelesscautiousin
eventhousandsoftrials(e.g.,Ratcliff&Rouder,1998).Naturally, thespeedcondition.Atthetimewhenpeoplewereinstructedto
thebehaviorvariesacrossthesetrials,andthesequentialsampling respondquickly oraccurately intheupcoming trial,the authors
model‘‘explains’’thisbythevariabilityofthestochasticprocess recorded the fMRI signal in the pre-supplementary motor area
of sampling information. However, we argue that the variability (pre-SMA)andthecaudatenucleus(CN).Theyreportedanegative
in behavior across trials could also be due and explained by the correlation between the difference in response caution in the
variabilityofmodelparameterssuchasadecisionthreshold(see speed and accuracy conditions and the difference in fMRI signal
alsoCraigmile,Peruggia,&VanZandt,2010andZandbelt,Purcell, inpre-SMAandCN.Theirconclusionwasthatahigherincreasein
Palmeri, Logan, & Schall, 2014). Moreover, behavioral variability baselineactivityinthepre-SMAandCNmediatedparticularlyfast
shouldinprinciplebereflectedinvariabilityoftherecordedbrain decisionsinthespeedcondition.
data(seealsoGlimcher,2005).Wesuggestthatthevariabilityof Forstmannetal.’s(2008)resultsclearlyindicateaninvolvement
behaviorandtheneuralsystemcouldbematchedtoeachotherand ofthepre-SMAandCNinresponsecaution.However,ifwecon-
might offer important insights about the neurocognitive system siderthepossibilitythatthedecisionthresholddoesnotstaycon-
andthemodelstryingtoexplainit. stantacrosstrialswithinaspecificcondition,thenthestudyleaves
Indeed,recentstudiesinmodel-basedcognitiveneuroscience openwhetherthesebrainstructurescouldalsobeimportantfor
thresholdadjustmentsonatrial-by-triallevel.Thefollowinghypo-
havetriedtoexploitthistrial-by-trialvariabilitybyfirstcontrast-
theticalexampleillustratesthepoint:Theaverage‘‘brainactivity’’
ingamodel’saveragepredictionswiththeparticipants’actualbe-
(acrossalltrialsandsubjects)inthepre-SMA/CNcouldbex =10
havior in single trials (i.e., response times [RTs] and/or choices) a
intheaccuracyconditionandx = 20inthespeedcondition;the
andthenusingthisprediction–observationdiscrepancytoinform s
averagevalueofresponsecautionparametercouldbe π = 20in
the analysis of neuroimaging data (Gluth, Rieskamp, & Büchel, a
2012;Gluth,Sommer,Rieskamp,&Büchel,2015;vanMaanenetal.,
2011).Thecentralaimofthisreviewandtutorialistopresenta
generalizationofthisapproachtoillustrateitsapplicabilityto(al- 1 Ratcliff’sdriftdiffusionmodelassumesacross-trialvariabilityinthedriftrate,
most)anyformofcognitivemodeland(almost)anyformofneural startingpoint,andnon-decisiontime(Ratcliff,1978;Ratcliff&McKoon,2008).
orphysiologicaldata. However,themodelonlyestimatesthe‘‘average’’amountofacross-trialvariability
The rest of the article is structured as follows: In Section 2, anddoesnotspecifytheamountordirectionofvariabilityinsingletrials.Thus,
it makes no trial-by-trial predictions of how the parameter values vary. More
we summarize neuroimaging studies of human decision making
recentmodelingtechniqueshavebeendevelopedtoaddressthispoint(Turner,van
thatcapturedtrial-by-trialdeviationsfrommodelpredictionsto
Maanen,&Forstmann,2015;Wiecki,Sofer,&Frank,2013),whichwediscussin
analyzethebraindataasoutlinedabove;inSection3,weprovide moredetailinSection6.

106 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
Fig.1. Neuroimagingstudiestestingfortrial-by-trialvariabilityinmodelestimates.(A)Thesingle-trialLBAmodelestimatesthemostlikelyheightofthestartingpoint
distributionineachtrialbasedontrial-wiseRTsinperceptualtaskssuchastherandomdotmotionparadigm(left).Therelatedresponsecautionis(negatively)linkedto
fMRIsignalsinthepre-SMA(right,redcluster).(B)Inasequential,value-basedchoicetask(left),asequentialsamplingmodelpredictedthecumulativesumofprobability
(CSP)ofmakingachoiceafter1to6samples(middle).However,onlyonechoicepertrialcanbeobserved(theexampleshowstwopossibleobservationsat3or5samples).
Similarto(A),theCSPattheactualchoicepointapproximatesresponsecautionandisalso(negatively)linkedtopre-SMAactivation(right).(C)Inamemory-basedchoice
paradigm,subjectsfirstlearnedtoassociate(foodsnack)optionswithpositionsonthescreenandthenhadtochoosebetweentwooptionsbyrecallingtheseassociations
(left).Asequentialsamplingmodelwasusedtodescribedecisionsandtoderivetheposteriorprobabilityofrememberingasnackatthedecisionstage(middle).Inthis
example,snack#37isnotchoseneventhoughithasahighersubjectivevaluethansnack#1.Accordingly,thememoryprobabilityforsnack#37(Posteriorx )isdecreased
comparedtotheinitialestimate(Px )thatisbasedonmodelingalltrials.AhighervalueofPosteriorxislinkedtohigheractivationintherighthippocampusattheencoding
stage(right).(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
Source:(A)AdaptedfromvanMaanenetal.,2011,JournalofNeuroscience.(B)AdaptedfromGluthetal.,2012,JournalofNeuroscience.(C)AdaptedfromGluthetal.,2015,
Neuron.
theaccuracyconditionand π = 10inthespeedcondition.Thus, aggregatedatasuggests.Suchareversepatternmightbeunlikely,
s
inthisexamplethereisanegativerelationshipbetweenbrainac- butcannotberuledout.Furthermore,itispossiblethatsimplyno
tivationxandmodelparameter π ,andtheymightfurtherbeneg- pattern at the trial-by-trial level exists. Hence, it appears unjus-
atively correlated across subjects (as in Forstmann et al., 2008). tifiedtoinfercognitiveandneuralprocessesinsingletrialsfrom
Such‘‘results’’wouldindicatethatwheneveractivityinthepre- brain–behaviorcorrelationsacrosssubjects(cf.Robinson,1950).
SMA/CNishigh,responsecautionislow.Yet,itispossiblethaton van Maanen et al. (2011) have recently addressed the idea
atrial-by-triallevel,justtheoppositeistrue:Inonetrialwithin of trial-by-trial adjustments of response caution in perceptual
theaccuracyconditionwithaparticularlylowpre-SMA/CNacti- decisionmaking(Fig.1(A)).TheauthorsextendedtheLBAmodel
vationofx a,t=1 = 8,responsecautionwasactuallynotthathigh to allow for single-trial variability of the starting point (and
( π a,t=1 =18)comparedtoanothertrialwithastrongerbrainsig- the drift rate). To do this, they first fitted the standard LBA
nal(x a,t=2 = 12; π a,t=2 = 22).Similarly,withinthespeedcondi- to the entire RT distribution and then derived the maximum
tionresponsecautioncouldbehigherinatrialwithaparticularly likelihood estimates of starting points for each trial based on
strongpre-SMA/CNsignal(x a,t=1 = 22; π a,t=1 = 12)compared the trial-specific RTs. In other words, they asked: ‘‘Keeping all
toanothertrialwithweakeractivation(x a,t=2 = 18; π a,t=2 = 8). otherparametersfixed(tothevaluesderivedfromLBA),whatis
Inotherwords,withinconditionstherecouldbeapositivecorre- the most likely parameter value for the starting point in trial t
π
lationbetweenxand andnotthenegativecorrelationthatthe given the RT in t?’’ Intuitively, faster RTs should indicate higher

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 107
starting points, but RTs are also affected by other parameters behavior. This likelihood and the average ‘‘memory’’ parameter,
(i.e.,driftrate,decisionthreshold,non-decisiontime).Thesingle- whichservedastheprior,werethenusedtoderivetheposterior
trial starting point estimates were then correlated with single- estimate of the ‘‘memory’’ parameter for the particular snack S
trial estimates of the fMRI hemodynamic response as derived accordingtoBayes’stheorem.Theresultingprobabilitywasthen
from an independent component analysis approach (see Eichele usedinthefMRIdataanalysisasaparametricmodulator(Büchel,
et al., 2008). van Maanen et al. found that the trial-by-trial Holmes, Rees, & Friston, 1998) of the hemodynamic response
signalofanindependentcomponentcomprisingthepre-SMAwas functionduringencoding.Therationaleofthisanalysiswasthat
linkedtothetrial-wiseestimatesofstartingpoints.Interestingly, the model-based probability of remembering a choice option is
a component that included the CN was also related to starting similartothewell-establishedsubsequentmemoryeffect,which
points but in a different way from the pre-SMA (only in speed- refers to a stronger fMRI signal in the hippocampus for (later)
emphasis trials that directly followed accuracy-emphasis trials), recalledversusforgottenitemsatthetimewhenitemsarestored
illustratinghowsingle-trialanalysescanprovidenewinsightsinto in memory (Kim, 2011; Paller & Wagner, 2002). Indeed, Gluth
brain–behaviorrelationships.Thesingle-trialLBAbyvanMaanen etal.(2015)foundthattheirsnack-specificestimateofmemory
etal.wasalsousedincombinationwithEEGdatatolinkvariability strengthwascorrelatedwithactivationintherighthippocampus
inresponsecautionwiththeamplitudeofthecontingentnegative intheencodingphase,showingthatwecanmake(trial-by-trial)
variation (Boehm, van Maanen, Forstmann, & van Rijn, 2014) as inferences about related cognitive processes such as memory or
wellaswithfMRIdatatolinkvariabilityinthedriftratewithtuning attentiononthebasisofobservingandmodelingdecisions.
curvesintheprimaryvisualcortex(Hoetal.,2012).
Furtherevidenceforatrial-wiseassociationbetweenresponse
3. Ageneralframeworkforcapturingvariability
caution and activation in the pre-SMA and CN comes from an
fMRI study on the emergence of value-based decisions (Gluth
Inthissection,weintroduceageneralapproachforcapturing
et al., 2012) (Fig. 1(B)). In this study, participants had to decide
trial-by-trial variability by assuming variability of the values
whethertotakeorrejectafinancialofferofunknownvaluebased
of model parameters that represent specific mechanisms of
onsequentiallyrevealedprobabilisticcues.Asequentialsampling
the model. Our method is most similar to the approach we
model was developed to predict which decision was made and
followedinGluthetal.(2015),inwhichwedeterminedtrial-by-
how many cues were sampled. One step in the analysis of the
trial posterior estimates of model parameters using a Bayesian
fMRIdataaddressedwhetherthetrial-wisediscrepancybetween
approach. For better comprehensibility, we supplement the
thenumberofsampledcuespredictedbythebestmodelandthe
generalformalizationwithaconcreteexampletakenfromGluth
actualnumberofsampledcuesbytheparticipantswasforecasted
et al. (2012). As described above, in this study trial-by-trial
by fMRI signals at the onset of each trial. The authors found a
estimatesofresponsecautionwerebasedontheprobabilitythata
networkofbrainareasincludingthepre-SMAandCNtoberelated
choicehasbeenmadeuntilthetimepointatwhichitwasactually
to this discrepancy estimate, meaning that when activity in the
made (see middle panel of Fig. 1(B)). This is, however, just an
pre-SMA and CN was high at the beginning of a trial, subjects
approximation of the most likely decision threshold parameter,
tended to sample fewer cues (i.e., behaved less cautiously) than
and we show how to arrive at more precise (Bayesian optimal)
predicted by the model. Importantly, the discrepancy does not
trial-by-trialthresholdsinthistask.
reflectmerelythenumberofsampledcues,becausethemodel’s
Letusstartwiththegeneralframework.Wesupposethatafter
predictionsvariedfromtrialtotrialbasedonwhatinformationwas π
estimatingamodelM thatcontainsparameter westillobserve
conveyedbytherespectivesequencesofcues.
somevariabilityofbehavioracrossnearlyidenticalsituationsthat
Ofcourse,mappingtrial-by-trialfluctuationsinbehaviorwith
cannotbeexplainedbythemodel.Wesuggestthatthevariability
variabilityinparametervaluesofcognitivemodelsisnotrestricted
ofthebehaviorcouldbeexplainedbythemodelMifoneassumes
to a specific process such as response caution in sequential π
thatparameter isnotfixedbutalsovariesacrosstheseidentical
sampling models but can be generalized to various processes of
situations.Toexplainthevariabilityofbehaviorbyvariabilityof
othermodels.Forinstance,inarecentstudy,Gluthetal.(2015)
thisparameterweareinterestedinestimatingtheexactvalueof
again used the discrepancy between average model predictions π
foraspecificsituationortrial.Morespecifically,weareaiming
andsingle-trialbehaviortoinferthestrengthofepisodicmemory π
onthebasisofmemory-guideddecisions(Fig.1(C)).Whilelyingin
at deriving the value for s,t of subject s in trial t that has the
highest posterior probability given the data D at t (i.e., choice
theMRscanner,theirparticipantsfirsthadtolearnassociations
and/or response time at t) and the prior probability of different
between different food snacks and different positions on the π
valuesfor withrespecttothesubject’sdataD inalltrials.The
s
a
c
n
r
d
ee
p
n
a
(
r
e
ti
n
c
c
ip
o
a
d
n
in
ts
g p
h
h
a
a
d
se
t
)
o
. L
r
a
e
t
c
e
a
r
l
,
l
tw
th
o
e
lo
a
c
s
a
so
ti
c
o
i
n
a
s
te
w
d
e
s
r
n
e
a
h
ck
ig
s
hl
a
ig
n
h
d
te
t
d
o
, posteriorprobabilityP (π s,t |D s,t ) isgivenbyBayes s ’srule:
c ( m 2 h o 0 o d 1 o e 5 s l e ) t m t a h i o e lo d r e s e n l d e a d c t k o de t t h c h i e s e y io m p n r s n e e f b e m y rr o e m n d i e c a (d n d e s e c m i o s f a io n a n ds s p e h o q a f u s e t e n h ). t e ia G t l l a u s s t k a h m . e M p t l o in a r g l e . P π s,t |D s,t =  P P  D  D s, s t , | t π |π s, s t ,  t  × × P P (π (π s ) s ) d π s , (1)
specifically,themodelincludedamemoryparametertoaccount
whereP
(
D s,t
|π
s
)
isthelikelihoodofthedataofsattgiven
π
,and
for interindividual differences in the ability to recall the snacks (π) π
P is the prior probability of ’s value (over the entire time
s
inthedecisionphase.Critically,thisparameterreflectsaperson’s
series).Theintegralsignshowsthatonehastomarginalizeover
averagememorycapacityoveralldecisiontrialsanddoesnottell π
allpossiblevaluesof .Theprocedurecanthusbedividedinto
whetheraparticularsnackhasbeenrememberedorforgotten.On s
fivesteps:
the other hand, observing that a person does not choose a very
goodsnack(onseveraloccasions)indicatesthatthesubjectmight 1. EstimatethemodelMacrossalltrialsonthebasisoftheentire
θ
haveforgottenaboutit.Theauthorsfollowedthisreasoningand datasetD s togetthemostlikelyparameterset s forsubjects
π
usedaBayesianapproachtoderivetheposteriorprobabilitythat (whichincludesasingleestimateof s ).
π
asnackS hadbeenrememberedbasedonthechoicesandchoice 2. Define a reasonable range of parameter values for (if this
s
setsthatincludedS.Technically,thecognitivemodelwasre-run has not already been done in Step 1). For example, the value
over these choice sets under the assumptions that S had been of a parameter representing the probability of remembering
rememberedorforgottentoobtainthelikelihoodoftheobserved informationmustliebetween0and1.

108 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
Fig.2. Exampletrialandestimatedchoiceprobability.(A)Sequenceofsequentiallypresentedcues.(B)Choiceprobabilityforacceptandrejectdecisionsateachcuefor
Subject24basedonfittingthebestperformingsequentialsamplingmodelinGluthetal.(2012)toalltrialsofthissubject.Theprobabilitiessumupto1.Ascanbeseen,the
modelpredictsthatchoicesaftersamplingonlyoneortwocuesareveryunlikely.
(π) π
3. GetP forallpossiblevaluesof byre-runningmodelM notinformedaboutwhethertheirchoicewascorrectorincorrect.
s s
θ
overalltrials,keepingtheparameterset constantexceptfor Generation of cue sequences was randomized. The sequential
s
π
andextractingthemodel’soverallfits(i.e.,thelikelihoods). samplingmodeltrackedthedevelopmentofevidencefor/against
s
(π)
Although P is estimated on the basis of the data and the accepting the offer by means of the log-evidence. However, we
s
θ
estimated parameter set s , it will function as the prior for assumedthatthisestimateofevidencewasaffectedbyGaussian
(π)
Eq.(1).Therefore,P s hastobenormalizedsothatitsums noise,withtheamountofnoisebeingafreeparameter.Themodel
upto1. predicted a choice when an upper (accept) or a lower (reject)
4. Get P
(
D s,t
|π
s
)
for all possible values of
π
s by re-running the decisionthresholdwascrossed.Themodelincludedaparameterto
model M only over trial t and extracting the probability that accountforthedecreaseofthedecisionthresholdwithtime(which
M predictsD s,t correctly.P
(
D s,t
|π
s
)
willserveaslikelihoodin improved the model fit strongly), so that the decision threshold
Eq.(1)(andthusdoesnothavetobenormalized). actuallycomprisestwocomponents,theinitialthresholdandthe
π π
5. ComputeEq.(1)forallpossiblevaluesof s andtakethe s,t decrease. Here, we are interested in the initial threshold and in
thathasthehighestposteriorprobability. derivingtrial-by-trialestimatesforit(sowekeepthedecreaseof
Steps4and5havetoberepeatedforeverytrial(andeverysubject) thethresholdandthenoiseparameterconstant).
to get the entire series of trial-by-trial variability in π . These Inourexample,wefocusonasingletrial(#103)ofarealsubject
valuescanthenbesubjectedtovariousneuroimaginganalyses;for (#24) from the study who behaved close to the group average
instance,theymightbeimplementedasaparametricmodulatorin with respect to how many cues were sampled (group average:
agenerallinearmodelanalysisoffMRIdata(seealsoSection6).
4.16±0.70SD;Subject24:4.25).Thecuesfrom1to6inthistrial
Asapracticalexample,wenowapplythisgeneralframework
were+2,−2,−2,+1,+1,and+1,soaftersuggestingaccepting
toderivetrial-by-trialestimatesofthedecisionthresholdofthe theofferatCue1theevidenceswitchedtorejectingatCues2and
sequential sampling model proposed by Gluth et al. (2012). As 3 and then turned back slowly to accepting at Cue 6 (Fig. 2(A)).
statedabove,thismodelpredictshowmanypiecesofinformationa Thesubjectrejectedtheofferafterseeingthefirstthreecues.To
subjectsamplesbeforebuyingorrejectinganoffer.Fig.2(A)shows illustrate the estimation of trial-by-trial variability, we imagine
anexampletrialoftheexperiment.Ineachtrial,thesubjectcould that this trial was repeated two times later in the experiment,
sampleuptosixcuestodecidewhethertoacceptortorejecta butnowthesubjectdidnotrejectafterthreecuesbutaccepted
stockoffer.Dependingontheperformanceofthestockthesubject the offer after one observed cue or after six observed cues. We
receivedapositiveornegativepayoff.Possiblecuevalueswere+2, nowexaminehowtheposteriorestimateofthethresholdchanges
+1, −1, and −2, where positive (negative) values increased the acrossthesethreetrials.
θ
probabilitythatacceptingthecurrentofferleadstoagain(loss)of As mentioned, the first step is to estimate parameter set s
money.Cuesappeared/disappearedconsecutivelyafter2–4suntil basedonalltrials.Thebestfitting(initial)thresholdvalueis1.729
a decision was made, and acquiring a cue led to a fixed amount (unlikeintheoriginalstudy,weuseasinglevalueforacceptand
ofinformationcosts.Ifachoicewasmadebeforepresentingthe rejectthresholdsforthesakeofsimplicity).Fig.2(B)presentsthe
lastcue,theremainingcueswerenotrevealed.Participantswere predictionsofthemodelforourtrialofinterest.Thesepredictions

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 109
Fig.3. Estimationoftheposteriorprobabilityoftheinitialthresholdvaluesintheexampletrialwith3possibleobservations.(A)Priorprobabilityofpossibleinitial
thresholdvaluesaccordingtofittingthemodeltoalltrials.(B)Likelihoodofinitialthresholdvaluesintheexampletrialgiveneachobservation.(C)Posteriorprobabilityof
initialthresholdvalues(modesarelabeled).Notably,themodesofthetwolikelyobservations(acceptat6,rejectat3)areclosetotheestimatedthresholdof1.729(indicated
bythedashedline),whereasthemodefortheunlikelydecisionafter1cueliesfarofftotheleftside.Thenarrowerrangeofthex-axisin(C)ischosentohighlightdifferences
betweenthethreedistributions.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
arederivedbyestimatingtheprobabilitythatthedecisionvariable possiblethresholdvalues(i.e.,P ( D |π ) ).2 Fig.3(B)depictsthese
s,t s
(i.e.,thesampledlog-evidence)crosseseitherthebuyorthereject likelihoodsforacceptingafter6(darkblue),acceptingafter1(light
blue),andrejectingafter3(red)cues.Asexpected,laterchoices
| thresholdafter | 1to 6cues, | withthe | thresholdsdecreasing |     | from |     |     |     |     |     |     |
| -------------- | ---------- | ------- | -------------------- | --- | ---- | --- | --- | --- | --- | --- | --- |
cue to cue (see Gluth et al., 2012, for details). Interestingly, an are associated with higher thresholds. Notably, the likelihood is
acceptdecisionafter6cuesisdeemedmorelikelythanwhatthe maximalformanylow(high)thresholdsinthecaseofacceptingat
subjectactuallydid(rejectedafter3cues),indicatingthesubject thebeginning(end),becauseanyofthesethresholdspredictthat
mighthavebehavedlesscautiouslyinthisparticulartrialthanon achoiceismadeforsureatthefirst(last)cue.Hence,uncertainty
average. aboutthesinglemostlikelythresholdvalueremainsinsuchcases.
InStep2,wespecifytherangebetween0.001and3(withastep This problem dissolves after applying Bayes’s rule as given by
|     |     |     |     |     |     | Eq. (1) (Step | 5), where | priors and | likelihoods | are | combined |
| --- | --- | --- | --- | --- | --- | ------------- | --------- | ---------- | ----------- | --- | -------- |
sizeof0.001)asourrangeofreasonable,possibleestimatesofthe
(Fig.3(C)).Thefactthatprobablethresholdvaluesarerestricted
thresholdparameter.Thisissomewhatarbitrary,butathreshold
|     |     |     |     |     |     | to a very | narrow range | is determined | by the | narrow | range of |
| --- | --- | --- | --- | --- | --- | --------- | ------------ | ------------- | ------ | ------ | -------- |
of0orlowermakeslittlesensebecausethemodelwouldalways
priors(seeFig.3(A)).Nevertheless,thepeakvaluesdifferbetween
predictimmediateresponses,andwithathresholdofhigherthan
3(whilekeepingallotherparametersconstant)themodelwould the three possible observations, showing that our method can
alwayspredictresponsesatthelastcue.Forthisparameterrange, effectively capture trial-by-trial variability with the parameter
themodelisrunagainoveralltrialstoobtainthelikelihoodvalues (only the relative and not the absolute distance of peak values
|     |     |     |     |     |     | matters). | If the peak values | should | all be the | same, | one could |
| --- | --- | --- | --- | --- | --- | --------- | ------------------ | ------ | ---------- | ----- | --------- |
ateachpossiblethreshold(Step3).Becausetheselikelihoodswill
|         |              |             |              |        | (π)           | increasethegranularityofthesearchforpossibleparametervalues |     |     |     |     |     |
| ------- | ------------ | ----------- | ------------ | ------ | ------------- | ----------------------------------------------------------- | --- | --- | --- | --- | --- |
| be used | as the prior | probability | of threshold | values | (i.e., P s ), |                                                             |     |     |     |     |     |
inSteps2to4(e.g.,usingastepsizeof0.0001orsmaller).
| they must   | be normalized | so that | they    | sum to 1 (see | above). As |     |     |     |     |     |     |
| ----------- | ------------- | ------- | ------- | ------------- | ---------- | --- | --- | --- | --- | --- | --- |
| can be seen | in Fig. 3(A), | there   | is only | a very narrow | range of   |     |     |     |     |     |     |
probablevaluesforthethresholdparameter.Re-runningthemodel
| is again | used in Step | 4, but now | this is | restricted to | our trial of | 2   |     |     |     |     |     |
| -------- | ------------ | ---------- | ------- | ------------- | ------------ | --- | --- | --- | --- | --- | --- |
Actually,Step4canberealizedduringStep3bysavingeachlikelihoodvalue
interesttoobtainthelikelihoodofthecurrentbehaviorgiventhe
pertrialandpossiblethresholdvalue.

110 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
Fig.4. Comparisonoftrial-by-trialthresholdsandchoicepoints.(A)Developmentofthresholdvalues(upperpanel)andchoicepoints(lowerpanel)overthecourseofthe
experiment.(B)Asexpected,thereisasubstantialbutnotperfectcorrelationbetweenthesetwovariables.
This procedure has to be done for all trials to obtain the inbehavior.Thusarecoverystudyisabletoevaluatewhethera
entire time course of trial-by-trial variability in the experiment. cognitivemodelandaplannedexperimentaresuitedtoidentify
thedatageneratingtrial-by-trialvariability.
| Fig. 4(A) | shows | the posterior | estimates |     | of thresholds | together |     |     |     |     |     |     |     |     |
| --------- | ----- | ------------- | --------- | --- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
with the choice points for all trials of Subject 24. A comparison Here,weagainusethemodelandexperimentfromGluthetal.
of thresholds and choice points is important because on the (2012) to exemplify this parameter variability recovery test. In
one hand there should be a positive relationship between them a first step, we generate variability in the threshold parameter
(earlierchoicesshouldimplylowerthresholds),butontheother from trial to trial, simulate the model, and apply our technique
hand a perfect correlation is undesirable, as it would render to test whether this variability can be recovered. We then keep
the trial-by-trial estimation of thresholds redundant. In fact, the the threshold parameter constant but generate variability in
correlationofthresholdsandchoicepointsforthissubjectis.58,so another parameter, the noise parameter of the accumulation
thereisasubstantialdependencybutalsosufficientunexplained process,toexaminetowhatextentthis‘‘noise’’variabilitytends
tobemisattributedto‘‘threshold’’variability.Finally,wegenerate
| variance | (implying | that thresholds |     | and | choice | points | may be |     |     |     |     |     |     |     |
| -------- | --------- | --------------- | --- | --- | ------ | ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
variabilityinbothparameterstoinvestigatepotentialinteractions
| differently | related  | to the   | neuroimaging | data).    | This    | is reasonable |     |         |                    |     |          |                  |     |        |
| ----------- | -------- | -------- | ------------ | --------- | ------- | ------------- | --- | ------- | ------------------ | --- | -------- | ---------------- | --- | ------ |
|             |          |          |              |           |         |               |     | between | them (for example, | it  | could be | that variability |     | in the |
| because     | choosing | early or | late in      | this task | heavily | depends       | on  |         |                    |     |          |                  |     |        |
thresholdcanonlyberecoveredaslongasvariabilityinthenoise
| how congruent |     | the cues | are: If the | cues | are all | positive | or all |           |              |              |           |     |          |       |
| ------------- | --- | -------- | ----------- | ---- | ------- | -------- | ------ | --------- | ------------ | ------------ | --------- | --- | -------- | ----- |
|               |     |          |             |      |         |          |        | parameter | is low). For | all recovery | analyses, | we  | took the | model |
negative,thesubjectmightdecideearlyevenwitharelativelyhigh
|            |                |     |          |            |           |     |          | parameters | of all 26 subjects | from | Gluth | et al., | added trial-wise |     |
| ---------- | -------------- | --- | -------- | ---------- | --------- | --- | -------- | ---------- | ------------------ | ---- | ----- | ------- | ---------------- | --- |
| threshold; | if consecutive |     | cues are | constantly | canceling |     | out, the |            |                    |      |       |         |                  |     |
subjectmightdecidelateevenwitharatherlowthreshold.Looking variability (i.e., Gaussian noise) to the respective parameter(s),
atthetimecourseisalsohelpfultoseewhetherthereisenough generated synthetic data for the 120 trials of each participant,
varianceinitself(ifnot,onecouldtryoutasmallerstepsize)as fittedthemodeltothesedata,andappliedthetechniquedescribed
wellastodetectoutliersandslowdriftsinthedevelopmentofthe in Section 3 to estimate trial-by-trial variability in the decision
|     |     |     |     |     |     |     |     | threshold. | By using the | parameters | of  | all subjects | we  | ensured |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------------ | ---------- | --- | ------------ | --- | ------- |
parameter.Outlierscoulddistorttheneuroimaginganalysis,and
|             |     |             |            |      |      |        |        | having reasonable | and | reasonably | different | parameter    |      | sets (cf. |
| ----------- | --- | ----------- | ---------- | ---- | ---- | ------ | ------ | ----------------- | --- | ---------- | --------- | ------------ | ---- | --------- |
| slow drifts | are | usually not | measurable | with | fMRI | or EEG | due to |                   |     |            |           |              |      |           |
|             |     |             |            |      |      |        |        | van Ravenzwaaij   | &   | Oberauer,  | 2009).    | The critical | test | is the    |
high-passfiltersandbaselinecorrections.
|     |     |     |     |     |     |     |     | strengthofthe   | correlationbetweentrueandestimated |             |     |             |                | trial-by- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ---------------------------------- | ----------- | --- | ----------- | -------------- | --------- |
|     |     |     |     |     |     |     |     | trial parameter | values                             | (van Maanen | et  | al., 2011). | To investigate |           |
4. Parameter(variability)recovery howtheamountofvariabilityaffectstherecoveryperformance,we
simulated10differentlevelsofvariability:thestandarddeviation
Fluctuationsinbehaviorarenotnecessarilyduetosystematic of the Gaussian distribution, which was used to create the trial-
changesintheneurocognitivesystembutmightjustberandom by-trialvariabilityinaparameter,rangedfrom10%to100%ofthe
| noise. For | instance, | if you | generate | simulations |     | of a sequential |     |     |     |     |     |     |     |     |
| ---------- | --------- | ------ | -------- | ----------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
standarddeviationofthisparameteracrosssubjects.Weran100
| sampling | model | with a | noisy evidence |     | accumulation |     | process, |     |     |     |     |     |     |     |
| -------- | ----- | ------ | -------------- | --- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
simulationspersubjectandvariabilitylevel.
youwillobservevariabilityinresponsetimesandaccuracyrates
|     |     |     |     |     |     |     |     | The first | step was | to test whether |     | our method | can | recover |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------- | --------------- | --- | ---------- | --- | ------- |
evenwithoutparametervariability.Apartfromthemoregeneral
|     |     |     |     |     |     |     |     | trial-by-trial | variability | in the | decision | threshold. | The left | panel |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | ------ | -------- | ---------- | -------- | ----- |
questionofwhetherparametervaluesarealwaysvariable(which of Fig. 5(A) shows the correlation between data-generating
wediscussinSection7),thisraisestheissueofparameterrecovery, and recovered parameters for a randomly selected subject and
orinourcaseof‘‘parametervariabilityrecovery’’:Giventhatthere simulation with the highest level of variability (i.e., 100% of the
.
is variability in the parameter of interest, does this variability group SD). The correlation is substantial (r = 0 65). However,
producesuch(fluctuating)behaviorthatweareabletocaptureit therangeofrecoveredparametersismuchsmallerthantherange
with our method? We think that researchers who seek to apply ofdata-generatingparameters,whichisduetothenarrowprior
our method must conduct parameter recovery tests (preferably distributions(seealsoFig.3(A)).TherightpanelofFig.5(A)depicts
beforedatacollection)toevaluatewhetheritispossibletoidentify the average correlations over all subjects and simulations as a
the variability in parameter values that might cause variability function ofthe level of parameter variability. Ifthe variability is

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 111
Fig.5. Parameterrecoveryresults.(A)Correlationbetweentrue(data-generating)andestimatedthresholdparametersforanexamplesubjectandsimulationwiththe
highestlevelofvariability(leftpanel)andaveragedacrossallsubjectsandsimulationsasafunctionofthelevelofvariability(rightpanel).Notably,thecorrelationsincrease
withincreasingparametervariability.Theintuitionforthiseffectisthattherehastobeenough(true)variabilityintheparametertogeneratesystematicchangesinbehavior,
whicharecapturedbytherecoveryanalysis.(B)Thesameasin(A)butforvariabilitygeneratedinthenoiseparameter.Thecorrelationintheexampleontheleftis−.23.
(C)Correlationsbetweentrueandestimatedthresholds(leftpanel)andbetweentruenoiseparametervaluesandestimatedthresholds(rightpanel),ifvariabilityisinduced
inbothparameters.Thecorrelationsareunaffectedbythedegreeofvariabilityintheotherparameter.
high enough, the correlations are sufficiently high (for the two Whenwedonotinducevariabilityinthethresholdbutinthe
highest levels of variability, more than 90% of the correlations noise parameter, this variability is picked up by our technique
areabove0.5).Takentogether,therangeofrecoveredparameters to some extent in form of a small negative correlation between
is reduced compared to the range of true parameters, but more the estimated value of the threshold parameter and the data-
importantlyrecoveredandtrueparametersarehighlycorrelated generating noise parameter (Fig. 5(B)). This is due to the fact
with each other, so that linking the variability estimate to thatsimulationswithhighnoiseinthedecisionvariabletendto
neuroimagingdataiswarranted. generatedecisionsthatareveryfast,whichisthenattributedto

112 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
alowdecisionthreshold(notethatthismightnothavebeenthe parameters-similartothepriordistributionsgeneratedinourap-
case, if we had modeled variability in the noise parameter, too). proach(Fig.2(A)).Infact,ourpracticalexampleaboveusedanap-
Importantly,thecorrelationseemstobesosmallthatsignificant proximationtothepriordistributionofthethresholdparameter
correlationswithneuroimagingdataarenottobeexpected.Thus, thatsuffersfromtwoshortcomings.First,thelikelihoodofthresh-
itisunlikelythat(potential)brainsignalsreflectingvariabilityin old values is calculated while keeping all other parameters con-
the noise parameter could be misattributed to variability in the stant (at the values taken from the initial estimation). A hierar-
thresholdparameter. chicalBayesianmodelingapproachcanhandlethismoreelegantly
The small negative correlation might further suggest that bysimultaneouslydrawingentireparametersetsduringthesam-
recoveryofthresholdvariabilitycouldbehamperedbyvariability pling process (but note that hierarchical modeling is also possi-
in the noise parameter. However, when variability is induced ble with maximum likelihood estimation; e.g., Farrell & Ludwig,
in both parameters, the correlation between true and estimated 2008).Thismightevenalleviatetheproblemofthereducedrange
thresholdparametersappearstobeunaffected,withtheaverage ofparametervariability,thatweobservedinourrecoveryanalyses
correlations (for high variability levels) being again above 0.6 (Fig.5(A)):Thisreductionresultsfromtheverynarrowpriordistri-
(Fig.5(C)).Thesameistrueforthenegativecorrelationbetween bution(Fig.3(A)),whichinturnisdeterminedbykeepingallother
thenoiseparameterandtheestimatedthresholds.Insummary,our parametersconstant.Second,wefollowedagrid-searchalgorithm
methodofestimatingtrial-by-trialvariabilityinmodelparameters withaspecificstepsizefortryingoutdifferentparametervalues,
tends to underestimate the variability on an absolute scale but whichrunstheriskofbeingtooimpreciseforcapturingvariabil-
recoversthe(relative)differencesfromtrialtotrial.Thus,itseems ityontheonehandandofrequiringmuchcomputationtimeon
to be suited for correlational analyses with neuroimaging data. theother.Thisbecomesirrelevantwithdirectlyestimatingthepa-
However,generalizingtheseresultstoothermodelsandtasksis rameterdistributionviaBayesianmodeling(thoughthisdifficulty
verylimited,andresearchersmustrunseparaterecoveryanalyses couldalsobereducedbyapplyingmoreefficientoptimizational-
fortheirownwork. gorithms).
Intriguingly,Bayesianmodelingcouldbeusedtoestimatetrial-
by-trialvariabilityinamodelparameterinasinglestep:Thehi-
5. Otherrequirementsandrecommendations
erarchicalBayesianmodelcouldassumeoneparametervaluefor
every trial that is drawn from a higher subject-level distribu-
In addition to running parameter recovery tests, some more
tion(justassubject-specificparametervaluescanbedrawnfrom
requirementswithrespecttobothbehavioralandbraindataand
higher group-level distributions with this technique; Kruschke,
thecognitivemodelshouldbefulfilled,ifonewantstoapplyour
2010andLee&Wagenmakers,2014).Thiswouldevenallowtest-
approachsuccessfully.Ideally,theresearchershouldtakecareof
ing whether the increased model complexity that comes with
thesearrangementsbeforeconductingthestudy,becauseatleast
assumingtrial-specificparameterestimatesiswarranted,forin-
some of them are irreversible. First, the experimental paradigm
stance, by comparing the trial-unspecific and the trial-specific
shouldallowresearcherstotrackbehavioralmeasuresduringfMRI
model by means of the deviance information criterion (Spiegel-
orEEGrecordingsforeverytrial.Thisseemstrivial,butespecially
halter, Best, Carlin, & van der Linde, 2002, 2014). Stated differ-
infMRIstudiesbehaviorisoftenrecordedinjustasmallportion
ently,onecouldtestwhetherassumingtrial-by-trialvariabilityin
of trials (e.g., Hindi Attar, Finckh, & Büchel, 2012) or only after
amodelparameterleadstoasubstantialimprovementofmodelfit
scanning(e.g.,Singeretal.,2004).Similarly,theneuroimagingset- (thatoutweighsthelossofparsimony).3 Despitealltheseadvan-
upmustallowcapturingsignaldifferencesonatrial-by-triallevel,
tagesofahierarchicalBayesianmodelingapproach,itisimportant
so positron emission tomography or fMRI block designs are not
tonotethatonedoesnothavetobeproficientinthesetechniques
suitableforourapproach.Thesameistrueforanalysesthatremove toapplyourmethod,butthatitalsoworkswithmaximumlikeli-
trial-specificvariabilitysuchasevent-relatedpotentialsofEEG. hoodestimationasshowninourexampleabove.
Besides identifying these requirements, we also give some Our final recommendation pertains to the quality of the
recommendations to facilitate the application and increase its cognitive model itself. The model should provide a very good
efficiency.Ourfirstpieceofadviceisthatarichsetofbehavioral descriptionofthedata,sothatremainingvariabilitycanbeclearly
data is helpful when one wants to explain variability across attributedtospecificcomponentsofitandnotjusttoitsgeneral
trials: If model predictions are based not only on choices but insufficiency. Our approach is only optimal within the limits
alsoonresponsetimes(suchasinsequentialsamplingmodels), of the underlying model. If the model itself makes little sense,
therearemoreopportunitiestoobservetrial-by-trialfluctuations thenattributingbehavioralfluctuationstooneofitscomponents
in behavior. This does not necessarily imply more trial-by-trial cannotbeverymeaningful,either.Hence,itisadvisabletosubject
variabilityinparameters,butgiventhatsuchvariabilitydoesexist, multiplemodelstoarigorousmodelcomparisonprocedurebefore
the likelihood of capturing it is increased by including response extracting trial-by-trial predictions on the basis of the winning
timesintheanalyses.Inperceptualdecisionmaking,forinstance, model.
peopleoftenachieveveryhighaccuracyratesofabove.9,which
meansthatforabout90%ofthetrialsparametervariabilitycould 6. Comparisonwithothertrial-by-trialtechniques
not be estimated, if a model takes only accuracy into account.
Alternatively, one could arrive at more precise predictions of How is our approach related to the studies and techniques
variabilitybycombiningmultiplechoicestospecifytheposterior discussed in Section 2 and to alternative approaches? As stated
distribution. For example, in Gluth et al. (2015) the probability above, the approach is basically a generalization of the method
of successful encoding of a single item was based on three used in Gluth et al. (2015), where memory-based choices were
observationsinthedecisionphase(Fig.1(B)). usedtoderivetheposteriorprobabilitythatachoiceoptionwas
Second, a Bayesian modeling approach (Lee & Wagenmakers,
2014) is highly recommended because it directly addresses and
solvesseveralpracticalissuesofourapproach:Itforcesthemod-
3 We would therefore recommend testing for trial-specific parameter values
elertospecifytherangeofpossibleparametervaluesbeforehand
onlyifthereisastronghypothesisthatthisvariabilityactuallyexistsandcauses
(thoughthisisusuallyalsodoneformaximumlikelihoodestima-
fluctuationsinbehavior.Note,however,thatimprovingamodel’spredictionof
tion),anditcandirectlyprovideposteriordistributionsofmodel behaviorisnottheprimarygoalofourapproach.

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 113
remembered. It is also similar to the methods used in the two model) and model comparisons determine whether adding the
other studies (Gluth et al., 2012; van Maanen et al., 2011)—but neural term to the model is warranted. The central difference
with some notable differences. Compared to Gluth et al. (2012), from our approach is that the brain signal has to be extracted
firsttoimplementitinthemodelthatpredictsbehavior(instead
| the new  | framework      | has       | a stronger | theoretical  |       | foundation, | as it |                     |         |                |     |             |       |            |            |
| -------- | -------------- | --------- | ---------- | ------------ | ----- | ----------- | ----- | ------------------- | ------- | -------------- | --- | ----------- | ----- | ---------- | ---------- |
|          |                |           |            |              |       |             |       | of first extracting |         | trial-by-trial |     | predictions | from  | the        | behavioral |
| provides | trial-by-trial | estimates |            | for distinct | model | components. |       |                     |         |                |     |             |       |            |            |
|          |                |           |            |              |       |             |       | model to            | predict | neuroimaging   |     | data,       | as in | our case). | Hence,     |
InGluthetal.(2012),thediscrepancybetweenmodelpredictions
|     |     |     |     |     |     |     |     | this method | requires |     | much | more | specific | hypotheses | about |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --- | ---- | ---- | -------- | ---------- | ----- |
andobservations(intermsoftooearlyortoolatepredictions)was
|     |     |     |     |     |     |     |     | which brain | signals | could | be  | related | to trial-by-trial |     | changes |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ----- | --- | ------- | ----------------- | --- | ------- |
usedtoapproximatefluctuationsintheinitialdecisionthreshold.
However,theseverygeneralmodelpredictionsarealsoaffected in cognitive processes: The researcher must have a good idea
by other parameters (i.e., the decrease of thresholds with time, about which kind of signal (e.g., theta-band power) in which
thestandarddeviationofthedriftrate).Ournewapproachallows brain region (e.g., medial prefrontal cortex) could be linked
isolating single model components (by holding the rest of the to which component (e.g., decision threshold) of the cognitive
modelconstant),whichalsoenablesacomparisonoftherespective model. Exploring the entire space of possible brain–behavior
contributions of different parameters to variability in the data relationshipswiththistechniquewouldbefartootimeconsuming.
|             |     |           |     |          |          |     |           | Furthermore, | the | software | tool | provided | by  | the research | group |
| ----------- | --- | --------- | --- | -------- | -------- | --- | --------- | ------------ | --- | -------- | ---- | -------- | --- | ------------ | ----- |
| (one simply | has | to repeat | the | analysis | outlined | in  | Section 3 |              |     |          |      |          |     |              |       |
(Wieckietal.,2013)isrestrictedtothedriftdiffusionmodel(and
| for other | parameters). |     | In van Maanen |     | et al. | (2011), | maximum |     |     |     |     |     |     |     |     |
| --------- | ------------ | --- | ------------- | --- | ------ | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
theLBA).
likelihoodestimatesforsingle-trialparameters(LBA’sstartpoints
Asimilarbutmoreadvancedmethodofcombiningneuroimag-
anddriftrates)werederivedanalytically,whichisdesirablebut
|                  |     |            |      |              |     |            |        | ing and | behavioral | data | in a | Bayesian | modeling | framework | was |
| ---------------- | --- | ---------- | ---- | ------------ | --- | ---------- | ------ | ------- | ---------- | ---- | ---- | -------- | -------- | --------- | --- |
| often infeasible |     | to derive. | More | importantly, |     | van Maanen | et al. |         |            |      |      |          |          |           |     |
introducedbyTurneretal.(2013).Theirideawastomodelboth
didnotimplementpriorsforparametervaluestoconstrainthese
|     |     |     |     |     |     |     |     | behavioral | and | brain data | sets | simultaneously |     | with | hierarchical |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---------- | ---- | -------------- | --- | ---- | ------------ |
maximumlikelihoodestimates.Thisstepisnotmandatorytoget
|     |     |     |     |     |     |     |     | Bayesian | modeling | to  | restrict | parameter | estimation |     | on both the |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --- | -------- | --------- | ---------- | --- | ----------- |
trial-by-trialpredictions,butithelpsavoidthatsomesingle-trial
‘‘neuralmodel’’andthe‘‘behavioralmodel’’.Acentraldifference
estimatesareveryextremeorequallylikely(e.g.,seethebluelines
fromFrank’sapproachisthatonedoesnothavetospecifywhich
inFig.3(B)).
|     |     |     |     |     |     |     |     | brain region | might | be  | related | to which | model | parameter | in ad- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | --- | ------- | -------- | ----- | --------- | ------ |
TwoconceptuallyrelatedapproacheswereproposedbyRatcliff, vance.Instead,anentiresetofareas(potentiallycoveringtheen-
Philiastides, and Sajda (2009) and by Ide, Shenoy, Yu, and Li tirebrain)canbeusedsimultaneously,andtheirrelationshiptothe
(2013),respectively.Usingdatafromaperceptualdecisionmaking modeliscomputedforeveryparameter.Inaddition,thejointmod-
task, Ratcliff et al. linked variability in the drift rate of the drift elingofbrainandbehavioraldatamakesitstatisticallyreciprocal;
diffusionmodel(DDM)toanEEGcomponentforwhichsingle-trial thatis,theneuraldatacanconstrainparameterestimationsofthe
estimatescouldbeobtained.Trialsweresplitonthebasisofthe behavioralmodelandviceversa.Thismethodhasrecentlybeen
amplitudeoftheEEGcomponent,andtheDDMwasestimatedfor appliedtoimprovepredictionsofRatcliff’sdriftdiffusionmodelfor
eachhalfofthetrialsseparately.Theauthorswereabletoshow aperceptualdecision-makingtaskbymeansofaddinginformation
thatthedriftratewashigherforhigh-amplitudecomparedtolow- fromanfMRIdataset(Turneretal.,2015).Theapproachisasgen-
amplitudetrialsevenwithinsingleconditionsofperceptualnoise. eralasoursandhastheadvantageofstatisticalreciprocity,butits
Themaindifferencetoourapproachisthattheneuraldatawas complexitymightdiscouragemanycognitiveneuroscientistsfrom
usingitfortheirmodelsanddatasets.Incontrast,ourmethodcan
| used to | inform | the behavioral | modeling, |     | whereas | our | approach |     |     |     |     |     |     |     |     |
| ------- | ------ | -------------- | --------- | --- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
berealizedquiteeasilybysimplyrestructuringformulasthatare
| relies on | directly | extracting | variability |     | via behavioral |     | modeling |     |     |     |     |     |     |     |     |
| --------- | -------- | ---------- | ----------- | --- | -------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
alreadypartofthecognitivemodel.Inotherwords,everyonewho
| and then | informing | the | neuroimaging |     | analysis. | Ide et | al. used a |     |     |     |     |     |     |     |     |
| -------- | --------- | --- | ------------ | --- | --------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
hasmanagedtosetupandfitacognitivemodelshouldbeableto
dynamicbeliefupdatingmodeltoestimatetrial-by-trialchanges
gothisonestepfurtherandapplyourtechnique.
| in the expectancy |     | of stop | signals | in a | stop-signal | detection | task |     |     |     |     |     |     |     |     |
| ----------------- | --- | ------- | ------- | ---- | ----------- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
andtorelatethistotheneuralencodingofexpectancyandsurprise
signalsintheanteriorcingulatecortex(seealsoHarléetal.,2015, 7. Potentialfindingsandwhattheymean
| who relates | this | model | to substance |     | abuse | in adolescents). | In  |     |     |     |     |     |     |     |     |
| ----------- | ---- | ----- | ------------ | --- | ----- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
contrasttoourapproach,theactualbehaviorofthesubjectsdid In this section, we present three scenarios of how trial-by-
notinformtheestimatesofbeliefupdating,butthemodelsimply trial variability in behavior might be explained by variability in
assumed that all subjects updated their expectancy in the same model parameters and related to neuroimaging data. We also
|     |     |     |     |     |     |     |     | discuss | implications | for | theoretical |     | developments |     | in cognitive |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ----------- | --- | ------------ | --- | ------------ |
manner.
|     |     |     |     |     |     |     |     | neuroscience. | We  | try | to keep | the discussion |     | at a | high level of |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | ------- | -------------- | --- | ---- | ------------- |
Anotherdifferentapproachtorelatingtrial-by-trialvariability
|     |     |     |     |     |     |     |     | generalizability |     | but we | complement |     | it with | real examples | from |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------ | ---------- | --- | ------- | ------------- | ---- |
incognitiveandneuralprocesseswasproposedbyMichaelFrank
|     |     |     |     |     |     |     |     | the literature. |     | All scenarios | are | based | on the | (methodologically |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------------- | --- | ----- | ------ | ----------------- | --- |
andcolleagues(Cavanaghetal.,2011;Cavanagh,Wiecki,Kochar,
simplebutstraightforward)ideathatthetrial-by-trialparameter
| & Frank,     | 2014;    | Frank   | et al., 2015; | Wiecki            | et             | al., 2013).     | Their     |                |           |         |                 |       |            |                  |             |
| ------------ | -------- | ------- | ------------- | ----------------- | -------------- | --------------- | --------- | -------------- | --------- | ------- | --------------- | ----- | ---------- | ---------------- | ----------- |
|              |          |         |               |                   |                |                 |           | variability,   | derived   | by      | applying        | our   | technique, | is               | used as a   |
| technique    | is based | on      | implementing  |                   | trial-by-trial | changes         | in        |                |           |         |                 |       |            |                  |             |
|              |          |         |               |                   |                |                 |           | predictor      | in a      | general | linear          | model | (GLM) of   | the neuroimaging |             |
| neuroimaging | data,    | such    | as variable   | peaks             | in             | the hemodynamic |           |                |           |         |                 |       |            |                  |             |
|              |          |         |               |                   |                |                 |           | data analysis. |           | The GLM | is particularly |       | popular    | in               | fMRI; here, |
| response     | function | of fMRI | data          | or trial-specific |                | power           | estimates |                |           |         |                 |       |            |                  |             |
|              |          |         |               |                   |                |                 |           | trial-by-trial | parameter |         | variability     | could | be         | implemented      | as a        |
takenfromEEGtime–frequencyanalyses,inhierarchicalBayesian
parametricmodulator:Itpredictshowthehemodynamicresponse
modelingofRatcliff’sdriftdiffusionmodel(Ratcliff,1978;Wiecki
|     |     |     |     |     |     |     |     | of the fMRI | blood–oxygen-level-dependent |     |     |     | signal | is  | modulated |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------------------------- | --- | --- | --- | ------ | --- | --------- |
et al., 2013). For example, Cavanagh et al. (2011) showed that as a function of the parameter variability at specific time points
thedecisionthresholdcanbeaffectedbyfluctuationsintheEEG intheexperiment(formoreinformation,seeBücheletal.,1998;
power of theta-band frequencies (4–8 Hz) at frontal electrodes, Büchel,Wise,Mummery,Poline,&Friston,1996;O’Dohertyetal.,
presumably reflecting an adaptive conflict-control mechanism 2007).Thesespecifictimepointscould,forinstance,refertowhen
mediatedbyinteractionsofthemedialprefrontalcortexwiththe decisionsaremadeorwhensubjectsareinstructedtoemphasize
subthalamicnucleus.Practically,thesupposedlyrelevantpartof speed or accuracy in the upcoming trial. There are, of course,
the neuroimaging data is first extracted on a trial-by-trial basis many other possible ways of linking parameter variability to
and then implemented in the hierarchical drift diffusion model neuroimagingdata(seealsoTurner,Forstmann,Love,Palmeri,&
using a regression approach. Posterior probability estimates of vanMaanen,2016),butanin-depthdiscussionisbeyondthescope
| the regression |     | coefficients | (that | relate | the brain | data | with the | ofthisarticle. |     |     |     |     |     |     |     |
| -------------- | --- | ------------ | ----- | ------ | --------- | ---- | -------- | -------------- | --- | --- | --- | --- | --- | --- | --- |

114 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
|           |             |            |     |           | π   |              |     | evidenceaccumulationprocess,reflectedinthegradualincrease |          |          |       |      |        |         |               |
| --------- | ----------- | ---------- | --- | --------- | --- | ------------ | --- | --------------------------------------------------------- | -------- | -------- | ----- | ---- | ------ | ------- | ------------- |
| Scenario1 | (Converging | Evidence). |     | Parameter |     | and activity | in  |                                                           |          |          |       |      |        |         |               |
|           |             |            |     |           |     |              |     | of firing                                                 | rates in | cortical | areas | such | as the | lateral | intraparietal |
brainregionxarecorrelatedacrosssubjects;applyingthetrial-by-
π
trial analysis proposed in this paper reveals that parameter is cortexandthefrontaleyefields,stopsatastereotypedlevelthatis
similarlyrelatedtobrainregionxonthesingle-triallevel. independentofresponsetimesandtaskdifficulty(Hanes&Schall,
|            |          |                |     |          |      |       |         | 1996; Roitman |     | & Shadlen, | 2002). | This | has | led to the | proposal |
| ---------- | -------- | -------------- | --- | -------- | ---- | ----- | ------- | ------------- | --- | ---------- | ------ | ---- | --- | ---------- | -------- |
| This first | scenario | is essentially |     | what has | been | found | for the |               |     |            |        |      |     |            |          |
thatadaptationsinresponsecautionaremediatedbyachangein
| association | of response | caution       | with  | activation |            | in the pre-SMA |        |           |          |              |     |        |      |                  |     |
| ----------- | ----------- | ------------- | ----- | ---------- | ---------- | -------------- | ------ | --------- | -------- | ------------ | --- | ------ | ---- | ---------------- | --- |
|             |             |               |       |            |            |                |        | the start | point of | accumulation |     | rather | than | by an adjustment | of  |
| and CN      | (Forstmann  | et al., 2008; | Gluth | et         | al., 2012; | van            | Maanen |           |          |              |     |        |      |                  |     |
thedecisionthresholditself(Bogacz,Wagenmakers,Forstmann,&
etal.,2011)(seeSection2),anditcanberegardedasmostdesired
Nieuwenhuis,2010;butseeSimen,2012).Consequently,looking
| and least | surprising. | Variability | across | and | within | subjects | in  |              |           |     |             |     |         |             |        |
| --------- | ----------- | ----------- | ------ | --- | ------ | -------- | --- | ------------ | --------- | --- | ----------- | --- | ------- | ----------- | ------ |
|           |             |             |        |     |        |          |     | for a neural | correlate |     | of response |     | caution | variability | at the |
(activationof)xisreflectedinvariabilityacrossandwithinsubjects
π time when the choice is made (i.e., when the decision variable
| in(behavior-basedestimationof) |           |              |     | .Importantly,thetrial-by-trial |     |     |          |          |                |           |       |             |     |             |            |
| ------------------------------ | --------- | ------------ | --- | ------------------------------ | --- | --- | -------- | -------- | -------------- | --------- | ----- | ----------- | --- | ----------- | ---------- |
|                                |           |              |     |                                |     |     |          | crosses  | the threshold) |           | might | not reveal  | any | significant | results.   |
| analysis                       | increases | the evidence | for | the brain–behavior             |     |     | link not |          |                |           |       |             |     |             |            |
|                                |           |              |     |                                |     |     |          | Instead, | the more       | promising |       | time window |     | to look     | at is when |
onlyquantitativelybutalsoqualitatively:AsdiscussedinSection2,
|                |     |                   |     |           |            |     |        | the decision | process |     | starts, or | even | earlier, | when | people are |
| -------------- | --- | ----------------- | --- | --------- | ---------- | --- | ------ | ------------ | ------- | --- | ---------- | ---- | -------- | ---- | ---------- |
| a relationship |     | between parameter |     | and brain | activation |     | across |              |         |     |            |      |          |      |            |
subjectsdoesnotnecessarilyimplythesamerelationshipwithin instructedorincentivizedtobehavemoreorlesscautiouslyinthe
subjects.Thetrial-by-trialanalysisisthereforemorethanamere upcomingtrial(Boehmetal.,2014;Forstmannetal.,2008;Gluth
replicationoftheinterindividualcorrelation. et al., 2012; Gluth, Rieskamp, & Büchel, 2013; Ivanoff, Branning,
|     |     |     |     |     |     |     |     | & Marois, | 2008; | van | Veen, Krug, | &   | Carter, | 2008). | Hence, not |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ----- | --- | ----------- | --- | ------- | ------ | ---------- |
Scenario2 (NewInsights).Parameter π andactivityinbrainregion findingarelationshipbetweentrial-by-trialvariabilityincognitive
xarecorrelatedacrosssubjects;applyingthetrial-by-trialanalysis modelcomponentsandneuroimagingdatamightsupportoreven
proposedinthispaperreveals,however,thatparameter π isnot promotetheoreticaldevelopmentsincognitiveneuroscience.
relatedtobrainregionxbuttoanotherbrainregionyonthesingle-
|     |     |     |     |     |     |     |     | Another | reason | for | not finding |     | an effect | of  | trial-by-trial |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | --- | ----------- | --- | --------- | --- | -------------- |
triallevel. variability in the neuroimaging data could be the low signal-
|     |     |     |     |     |     |     |     | to-noise | ratio of | the latter. | Both | fMRI | and EEG | analyses | require |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ----------- | ---- | ---- | ------- | -------- | ------- |
Thisscenarioisdefinitelylesslikely—butstillpossibleasshown
|         |                 |       |     |            |        |     |          | many repetitions |        | of trials | to      | separate    | the | faint hemodynamic |           |
| ------- | --------------- | ----- | --- | ---------- | ------ | --- | -------- | ---------------- | ------ | --------- | ------- | ----------- | --- | ----------------- | --------- |
| above.4 | One explanation | could | be  | that while | region | x   | sets the |                  |        |           |         |             |     |                   |           |
|         |                 |       |     |            |        |     |          | or electrical    | signal | from      | all the | differences |     | sources           | of random |
overall(i.e.,subject-orcondition-wise)levelofparameterp,region
yisresponsibleforthetrial-by-trialfine-tuning.Inthiscase,there fluctuations,includingmovements,physiologicalartifactssuchas
shouldbeafunctionalconnectionbetweenxandy,whichcould heartbeatandrespiration,orartifactsofthemeasurementtools
berevealedusingcouplinganalysessuchaspsychophysiological (Huettel, Song, & McCarthy, 2014; Luck, 2014). This might raise
interaction (Friston et al., 1997) or dynamic causal modeling skepticismaboutwhetherthesetechniquesareinanyformsuited
(Friston, Harrison, & Penny, 2003). A more in-depth analysis of tocapturethetrial-by-trialchangeswhichweareaddressinghere.
thebehavioraldatamayalsohelpclarifythiscomplexpatternof However,wewouldarguethatthereisnoreasontobelievethat
results. For instance, making an error often triggers adaptations the low signal-to-noise ratio of neuroimaging tools should be
onthenexttrial(Rabbitt&Rodgers,1977),includingchangesin moreproblematicforinvestigatingparametriceffectsascompared
responsecaution(Dutilhetal.,2012).Regionycouldthusbeabrain to simpler comparisons of experimental conditions. In fact, it
areaassociatedwitherrorprocessingandcognitivecontrol,such seems more likely that simple contrasts such as ‘‘task vs. rest’’
astheanteriorcingulatecortex(Holroyd&Coles,2002;Shenhav, or ‘‘emotional vs. neutral’’ in an fMRI study are systematically
Botvinick,&Cohen,2013),whichexertsanadditionalinfluenceon confoundedbyheartrateorrespiration.Havingsaidthis,wewould
responsecautionunderhighconflict(Botvinick,Nystrom,Fissell, recommendthatresearchersseekingtoapplyourmethodsshould
Carter,&Cohen,1999;Cavanaghetal.,2011).
|     |     |     |     |     |     |     |     | consider | (and maybe | even | test) | whether | the | targeted | trial-wise |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---- | ----- | ------- | --- | -------- | ---------- |
variationsinmodelparametersmightbecontaminatedby(regular
| Scenario3 | (InconsistentEvidence).Applyingthetrial-by-trialanal- |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
variationsin)physiologicalartifacts.
ysisproposedinthispaperdoesnotrevealanyrelationshipofpa-
| rameter  | π withtheneuroimagingdata. |        |         |       |          |        |      |               |     |     |     |     |     |     |     |
| -------- | -------------------------- | ------ | ------- | ----- | -------- | ------ | ---- | ------------- | --- | --- | --- | --- | --- | --- | --- |
|          |                            |        |         |       |          |        |      | 8. Conclusion |     |     |     |     |     |     |     |
| At first | glance,                    | such a | finding | could | be taken | simply | as a |               |     |     |     |     |     |     |     |
nullfindingwithoutanytheoreticalimplicationsatall.However,
|            |        |                  |             |                |           | π   |        | Wehaveintroducedageneral,comparativelysimpletechnique        |     |     |     |     |     |     |     |
| ---------- | ------ | ---------------- | ----------- | -------------- | --------- | --- | ------ | ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
| a lack of  | neural | support for      | variability | in             | parameter |     | might  |                                                              |     |     |     |     |     |     |     |
|            |        |                  |             |                |           | π   |        | tocapturetrial-by-trialvariabilityincognitivemodelparameters |     |     |     |     |     |     |     |
| be because | the    | model assumption |             | of variability |           | in  | is not |                                                              |     |     |     |     |     |     |     |
useful to describe the brain data (and thus might also not be (and similar components) to explain trial-by-trial variability in
useful to describe the behavioral data). As outlined in Section 4, behavior and link it to the analysis of neuroimaging data. Our
behavior is always variable because the neurocognitive system approach relies on deriving probability estimates for different
isalwaysaffectedby‘‘randomnoise’’.Applyingourtechniqueto parameter values over all trials as well as for each individual
a randomly fluctuating behavioral measure will attribute these trial and then combining these estimates by means of Bayes’s
fluctuations to the parameter of interest. Yet, it is possible that rule to obtain the most likely posterior estimate in every trial.
| this purported |     | parameter variability |     | does | not exist. | To  | give an |            |            |     |        |         |             |     |             |
| -------------- | --- | --------------------- | --- | ---- | ---------- | --- | ------- | ---------- | ---------- | --- | ------ | ------- | ----------- | --- | ----------- |
|                |     |                       |     |      |            |     |         | Crucially, | the method |     | can be | applied | to (almost) |     | any type of |
example,neurophysiologicalstudiesinmonkeyssuggestthatthe mathematicalmodelaswellastoabroadrangeofneuroimaging
dataandanalyses(butespeciallytoevent-relatedfMRIdataand
time–frequencyEEGanalyses).Inaddition,theapproachcanalso
beusedincombinationwithphysiologicalmeasuressuchasskin
4 Becausethisscenarioislesslikely,moreempiricalevidencewouldbeneededto
conductance,electromyography,heartrate,eyetracking,orpupil
establishthatπandyareindeedlinkedtoeachother.Forinstance,asmall-volume
correctionapproachtothemultiple-comparisonproblemforfMRI(Poldrack,2006) dilation. We hope to encourage other researchers in cognitive
canbeusedtoincreasethesensitivitytodetectarelationshipofπwithregionx
andsystemsneurosciencetoemployourtechniquetotightenthe
(becausexhasalreadybeenshowntobelinkedtoπacrosssubjects)butnotwith
linksbetweentheirneuroimagingdataandtheirfavoritecognitive
regiony(unlessthereisanother,strongtheoreticalpredictionforwhyactivityin
regionyshouldbeassociatedwithvariabilityinπ). modelstoallowfortheoryadvancement.

S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116 115
Acknowledgments
|     |     |     |     |     |     |     | Friston, K. | J., Buechel,        | C., Fink, | G. R., Morris, | J.,          | Rolls, E., | & Dolan, R. J. |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------------------- | --------- | -------------- | ------------ | ---------- | -------------- |
|     |     |     |     |     |     |     | (1997).     | Psychophysiological |           | and modulatory | interactions | in         | neuroimaging.  |
WethankAnitaToddforeditingthemanuscript.Thisresearch NeuroImage,6(3),218–229.http://doi.org/10.1006/nimg.1997.0291.
|               |     |            |      |     |       |                  | Friston, K. | J., Harrison, | L., & | Penny, W. D. | (2003). Dynamic | causal | modelling. |
| ------------- | --- | ---------- | ---- | --- | ----- | ---------------- | ----------- | ------------- | ----- | ------------ | --------------- | ------ | ---------- |
| was supported |     | by a grant | from | the | Swiss | National Science |             |               |       |              |                 |        |            |
NeuroImage,19(4),1273–1302.
Foundation(SNSFgrant100014_153616)tobothauthors. Garrison,J.,Erdeniz,B.,&Done,J.(2013).Predictionerrorinreinforcementlearning:
ameta-analysisofneuroimagingstudies.Neuroscience&BiobehavioralReviews,
37(7),1297–1310.http://doi.org/10.1016/j.neubiorev.2013.03.023.
References Gläscher, J., Hampton, A. N., & O’Doherty, J. P. (2009). Determining a role
|     |     |     |     |     |     |     | for ventromedial |     | prefrontal | cortex in encoding |     | action-based | value signals |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ---------- | ------------------ | --- | ------------ | ------------- |
Bartra, O., McGuire, J. T., & Kable, J. W. (2013). The valuation system: a during reward-related decision making. Cerebral Cortex, 19(2), 483–495.
coordinate-based meta-analysis of BOLD fMRI experiments examin- http://doi.org/10.1093/cercor/bhn098.
ing neural correlates of subjective value. NeuroImage, 76, 412–427. Glimcher,P.W.(2005).Indeterminacyinbrainandbehavior.AnnualReviewofPsy-
chology,56,25–56.http://doi.org/10.1146/annurev.psych.55.090902.141429.
http://doi.org/10.1016/j.neuroimage.2013.02.063.
Behrens,T.E.J.,Woolrich,M.W.,Walton,M.E.,&Rushworth,M.F.S.(2007). Gluth, S., Rieskamp, J., & Büchel, C. (2012). Deciding when to decide: time-
Learningthevalueofinformationinanuncertainworld.NatureNeuroscience, variant sequential sampling models explain the emergence of value-based
decisionsinthehumanbrain.JournalofNeuroscience,32(31),10686–10698.
10(9),1214–1221.http://doi.org/10.1038/nn1954.
Boehm,U.,vanMaanen,L.,Forstmann,B.,&vanRijn,H.(2014).Trial-by-trialfluc- http://doi.org/10.1523/JNEUROSCI.0727-12.2012.
tuationsinCNVamplitudereflectanticipatoryadjustmentofresponsecaution. Gluth, S., Rieskamp, J., & Büchel, C. (2013). Classic EEG motor potentials
NeuroImage,96,95–105.http://doi.org/10.1016/j.neuroimage.2014.03.063. track the emergence of value-based decisions. NeuroImage, 79, 394–403.
Bogacz,R.,Brown,E.,Moehlis,J.,Holmes,P.,&Cohen,J.D.(2006).Thephysics http://doi.org/10.1016/j.neuroimage.2013.05.005.
Gluth,S.,Rieskamp,J.,&Büchel,C.(2014).Neuralevidenceforadaptivestrategy
ofoptimaldecisionmaking:aformalanalysisofmodelsofperformancein
two-alternative forced-choice tasks. Psychological Review, 113(4), 700–765. selectioninvalue-baseddecision-making.CerebralCortex,24(8),2009–2021.
http://doi.org/10.1037/0033-295X.113.4.700. http://doi.org/10.1093/cercor/bht049.
Bogacz,R.,Wagenmakers,E.-J.,Forstmann,B.U.,&Nieuwenhuis,S.(2010).The Gluth, S., Sommer, T., Rieskamp, J., & Büchel, C. (2015). Effective connec-
neural basis of the speed-accuracy tradeoff. Trends in Neurosciences, 33(1), tivity between hippocampus and ventromedial prefrontal cortex con-
|     |     |     |     |     |     |     | trols preferential |     | choices | from memory. | Neuron, | 86(4), | 1078–1090. |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ------- | ------------ | ------- | ------ | ---------- |
10–16.http://doi.org/10.1016/j.tins.2009.09.002.
Boll, S., Gamer, M., Gluth, S., Finsterbusch, J., & Büchel, C. (2013). Separate http://doi.org/10.1016/j.neuron.2015.04.023.
amygdalasubregionssignalsurpriseandpredictivenessduringassociativefear Hanes,D.P.,&Schall,J.D.(1996).Neuralcontrolofvoluntarymovementinitiation.
Science,274(5286),427–430.
| learning | in humans. | The European | Journal | of  | Neuroscience, | 37(5), 758–767. |     |     |     |     |     |     |     |
| -------- | ---------- | ------------ | ------- | --- | ------------- | --------------- | --- | --- | --- | --- | --- | --- | --- |
http://doi.org/10.1111/ejn.12094. Harlé, K. M., Stewart, J. L., Zhang, S., Tapert, S. F., Yu, A. J., & Paulus, M. P.
(2015).Bayesianneuraladjustmentofinhibitorycontrolpredictsemergence
Botvinick,M.,Nystrom,L.E.,Fissell,K.,Carter,C.S.,&Cohen,J.D.(1999).Conflict
monitoring versus selection-for-action in anterior cingulate cortex. Nature, ofproblemstimulantuse.Brain,138(11),3413–3426.http://doi.org/10.1093/
| 402(6758),179–181.http://doi.org/10.1038/46035. |     |     |     |     |     |     | brain/awv246. |         |              |                |     |           |             |
| ----------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------- | ------- | ------------ | -------------- | --- | --------- | ----------- |
|                                                 |     |     |     |     |     |     | Hey, J. D.    | (1995). | Experimental | investigations |     | of errors | in decision |
Brandstätter,E.,Gigerenzer,G.,&Hertwig,R.(2006).Thepriorityheuristic:Making
choiceswithouttrade-offs.PsychologicalReview,113,409–432. making under risk. European Economic Review, 39(3–4), 633–640.
Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice http://doi.org/10.1016/0014-2921(09)40007-4.
response time: linear ballistic accumulation. Cognitive Psychology, 57(3), HindiAttar,C.,Finckh,B.,&Büchel,C.(2012).Theinfluenceofserotoninonfear
153–178.http://doi.org/10.1016/j.cogpsych.2007.12.002. learning.PLoSOne,7(8),e42397.http://doi.org/10.1371/journal.pone.0042397.
|     |     |     |     |     |     |     | Ho, T., Brown, | S., | van Maanen, | L., Forstmann, | B.  | U., Wagenmakers, | E.-J., |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ----------- | -------------- | --- | ---------------- | ------ |
Büchel,C.,Holmes,A.P.,Rees,G.,&Friston,K.J.(1998).Characterizingstimu-
lus–responsefunctionsusingnonlinearregressorsinparametricfMRIexperi- & Serences, J. T. (2012). The optimality of sensory processing during
ments.NeuroImage,8(2),140–148.http://doi.org/10.1006/nimg.1998.0351. the speed-accuracy tradeoff. Journal of Neuroscience, 32(23), 7992–8003.
Büchel, C., Wise, R. J., Mummery, C. J., Poline, J. B., & Friston, K. J. (1996). http://doi.org/10.1523/JNEUROSCI.0340-12.2012.
Nonlinearregressioninparametricactivationstudies.NeuroImage,4(1),60–66. Holroyd,C.B.,&Coles,M.G.H.(2002).Theneuralbasisofhumanerrorprocessing:
reinforcementlearning,dopamine,andtheerror-relatednegativity.Psycholog-
http://doi.org/10.1006/nimg.1996.0029.
Busemeyer, J. R., & Townsend, J. T. (1993). Decision field theory: a dynamic- icalReview,109(4),679–709.
cognitiveapproachtodecisionmakinginanuncertainenvironment.Psycho- Huettel,S.A.,Song,A.W.,&McCarthy,G.(2014).Functionalmagneticresonance
imaging(3rded.).Sunderland,Massachusetts,U.S.A:SinauerAssociates,Inc.,
logicalReview,100(3),432–459.
| Cavanagh,J.F.,Wiecki,T.V.,Cohen,M.X.,Figueroa,C.M.,Samanta,J.,Sherman,S.J., |     |     |     |     |     |     | Publishers. |     |     |     |     |     |     |
| --------------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- |
Ide,J.S.,Shenoy,P.,Yu,A.J.,&Li,C.R.(2013).Bayesianpredictionandevaluation
&Frank,M.J.(2011).Subthalamicnucleusstimulationreversesmediofrontal
influence over decision threshold. Nature Neuroscience, 14(11), 1462–1467. in the anterior cingulate cortex. Journal of Neuroscience, 33(5), 2039–2047.
http://doi.org/10.1523/JNEUROSCI.2201-12.2013.
http://doi.org/10.1038/nn.2925.
Cavanagh, J. F., Wiecki, T. V., Kochar, A., & Frank, M. J. (2014). Eye track- Ivanoff, J., Branning, P., & Marois, R. (2008). fMRI evidence for a dual process
ing and pupillometry are indicators of dissociable latent decision pro- account of the speed-accuracy tradeoff in decision-making. PloS One, 3(7),
e2635.http://doi.org/10.1371/journal.pone.0002635.
| cesses. | Journal | of Experimental | Psychology: |     | General, | 143(4), 1476–1488. |     |     |     |     |     |     |     |
| ------- | ------- | --------------- | ----------- | --- | -------- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
http://doi.org/10.1037/a0035813. Kahneman,D.,&Tversky,A.(1979).Prospecttheory:ananalysisofdecisionunder
risk.Econometrica,47(2),263–292.http://doi.org/10.2307/1914185.
Clithero,J.A.,&Rangel,A.(2014).Informaticparcellationofthenetworkinvolvedin
|     |     |     |     |     |     |     | Kim, H. (2011). | Neural | activity | that predicts | subsequent | memory | and forget- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ------ | -------- | ------------- | ---------- | ------ | ----------- |
thecomputationofsubjectivevalue.SocialCognitiveandAffectiveNeuroscience,
9(9),1289–1302.http://doi.org/10.1093/scan/nst106. ting: a meta-analysis of 74 fMRI studies. NeuroImage, 54(3), 2446–2461.
http://doi.org/10.1016/j.neuroimage.2010.09.045.
| Craigmile, | P. F., | Peruggia, | M., & | Van Zandt, | T.  | (2010). Hierarchical |     |     |     |     |     |     |     |
| ---------- | ------ | --------- | ----- | ---------- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
Bayes models for response time data. Psychometrika, 75(4), 613–632. Krugel, L. K., Biele, G., Mohr, P. N. C., Li, S.-C., & Heekeren, H. R. (2009).
|     |     |     |     |     |     |     | Genetic | variation | in dopaminergic | neuromodulation |     | influences | the abil- |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --------------- | --------------- | --- | ---------- | --------- |
http://doi.org/10.1007/s11336-010-9172-6.
|     |     |     |     |     |     |     | ity to | rapidly | and flexibly | adapt decisions. | Proceedings | of  | the National |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ------------ | ---------------- | ----------- | --- | ------------ |
Daw,N.D.,O’Doherty,J.P.,Dayan,P.,Seymour,B.,&Dolan,R.J.(2006).Cortical
substratesforexploratorydecisionsinhumans.Nature,441(7095),876–879. Academy of Sciences of the United States of America, 106(42), 17951–17956.
http://doi.org/10.1073/pnas.0905191106.
http://doi.org/10.1038/nature04766.
Kruschke,J.K.(2010).Whattobelieve:Bayesianmethodsfordataanalysis.Trends
| Dutilh, G., | Vandekerckhove, | J., | Forstmann, | B. U., | Keuleers, | E., Brysbaert, M., & |     |     |     |     |     |     |     |
| ----------- | --------------- | --- | ---------- | ------ | --------- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
Wagenmakers,E.-J.(2012).Testingtheoriesofpost-errorslowing.Attention, inCognitiveSciences,14(7),293–300.http://doi.org/10.1016/j.tics.2010.05.001.
Lee,M.D.,&Wagenmakers,E.-J.(2014).Bayesiancognitivemodelingapractical
| Perception, | & Psychophysics, |     | 74(2), 454–465. |     | http://doi.org/10.3758/s13414- |     |                                  |     |     |     |     |     |     |
| ----------- | ---------------- | --- | --------------- | --- | ------------------------------ | --- | -------------------------------- | --- | --- | --- | --- | --- | --- |
| 011-0243-2. |                  |     |                 |     |                                |     | course.CambridgeUniversityPress. |     |     |     |     |     |     |
Lewandowsky,S.,&Farrell,S.(2011).Computationalmodelingincognition:Principles
| Eichele, T., | Debener, | S., Calhoun, | V.  | D., Specht, | K., | Engel, A. K., Hug- |     |     |     |     |     |     |     |
| ------------ | -------- | ------------ | --- | ----------- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
andpractice.LosAngeles,Calif.:SagePublications.
dahl, K., et al. (2008). Prediction of human errors by maladaptive Loomes, G., & Sugden, R. (1995). Incorporating a stochastic element
| changes | in event-related |        | brain networks. |     | Proceedings | of the National     |               |     |           |                   |         |          |          |
| ------- | ---------------- | ------ | --------------- | --- | ----------- | ------------------- | ------------- | --- | --------- | ----------------- | ------- | -------- | -------- |
|         |                  |        |                 |     |             |                     | into decision |     | theories. | European Economic | Review, | 39(3–4), | 641–648. |
| Academy | of Sciences      | of the | United States   | of  | America,    | 105(16), 6173–6178. |               |     |           |                   |         |          |          |
http://doi.org/10.1016/0014-2921(94)00071-7.
http://doi.org/10.1073/pnas.0708965105. Luce,R.D.(1986).Responsetimes.NewYork;Oxford:OxfordUniversityPress;
Farrell,S.,&Ludwig,C.J.H.(2008).Bayesianandmaximumlikelihoodestimation
ClarendonPress.
ofhierarchicalresponsetimemodels.PsychonomicBulletin&Review,15(6), Luck,S.J.(2014).Anintroductiontotheevent-relatedpotentialtechnique(2nded.).
1209–1217.http://doi.org/10.3758/PBR.15.6.1209.
Cambridge,Massachusetts:TheMITPress.
Forstmann,B.U.,Dutilh,G.,Brown,S.,Neumann,J.,vonCramon,D.Y.,Ridderinkhof, McFadden, D. (2001). Economic choices. The American Economic Review, 91(3),
| K.R.,etal.(2008).Striatumandpre-SMAfacilitatedecision-makingundertime |     |     |     |     |     |     | 351–378. |     |     |     |     |     |     |
| --------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
pressure.ProceedingsoftheNationalAcademyofSciencesoftheUnitedStatesof McGuire,J.T.,Nassar,M.R.,Gold,J.I.,&Kable,J.W.(2014).Functionallydissociable
America,105(45),17538–17542.http://doi.org/10.1073/pnas.0805903105. influencesonlearningrateinadynamicenvironment.Neuron,84(4),870–881.
Frank, M. J., Gagne, C., Nyhus, E., Masters, S., Wiecki, T. V., Cavanagh, J. F., http://doi.org/10.1016/j.neuron.2014.10.013.
et al. (2015). fMRI and EEG predictors of dynamic decision parameters O’Doherty,J.P.,Dayan,P.,Schultz,J.,Deichmann,R.,Friston,K.,&Dolan,R.J.(2004).
duringhumanreinforcementlearning.JournalofNeuroscience,35(2),485–494. Dissociablerolesofventralanddorsalstriatumininstrumentalconditioning.
http://doi.org/10.1523/JNEUROSCI.2036-14.2015. Science,304(5669),452–454.http://doi.org/10.1126/science.1094285.

116 S.Gluth,J.Rieskamp/JournalofMathematicalPsychology76(2017)104–116
O’Doherty,J.P.,Hampton,A.,&Kim,H.(2007).Model-basedfMRIanditsapplication Singer,T.,Seymour,B.,O’Doherty,J.,Kaube,H.,Dolan,R.J.,&Frith,C.D.(2004).
torewardlearninganddecisionmaking.AnnalsoftheNewYorkAcademyof Empathyforpaininvolvestheaffectivebutnotsensorycomponentsofpain.
Sciences,1104,35–53.http://doi.org/10.1196/annals.1390.022. Science,303(5661),1157–1162.http://doi.org/10.1126/science.1093535.
Paller,K.A.,&Wagner,A.D.(2002).Observingthetransformationofexperience Smith,P.L.,&Ratcliff,R.(2004).Psychologyandneurobiologyofsimpledecisions.
intomemory.TrendsinCognitiveSciences,6(2),93–102. TrendsinNeurosciences,27(3),161–168.http://doi.org/10.1016/j.tins.2004.01.
Payzan-LeNestour,E.,Dunne,S.,Bossaerts,P.,&O’Doherty,J.P.(2013).Theneural 006.
representationofunexpecteduncertaintyduringvalue-baseddecisionmaking. Spiegelhalter,D.J.,Best,N.G.,Carlin,B.P.,&vanderLinde,A.(2002).Bayesian
Neuron,79(1),191–201.http://doi.org/10.1016/j.neuron.2013.04.037. measuresofmodelcomplexityandfit.JournaloftheRoyalStatisticalSociety.
Pessiglione,M.,Petrovic,P.,Daunizeau,J.,Palminteri,S.,Dolan,R.J.,&Frith,C. SeriesB(StatisticalMethodology),64(4),583–639.http://doi.org/10.1111/1467-
D.(2008).Subliminalinstrumentalconditioningdemonstratedinthehuman 9868.00353.
brain.Neuron,59(4),561–567.http://doi.org/10.1016/j.neuron.2008.07.005. Spiegelhalter,D.J.,Best,N.G.,Carlin,B.P.,&vanderLinde,A.(2014).Thedeviance
Poldrack,R.A.(2006).RegionofinterestanalysisforfMRI.SocialCognitiveand informationcriterion:12yearson.JournaloftheRoyalStatisticalSociety.Series
AffectiveNeuroscience,2(1),67–70.http://doi.org/10.1093/scan/nsm006. B(StatisticalMethodology),76(3),485–493.http://doi.org/10.1111/rssb.12062.
Turner, B. M., Forstmann, B. U., Love, B. C., Palmeri, T. J., & van Maanen, L.
Rabbitt,P.,&Rodgers,B.(1977).Whatdoesamandoafterhemakesanerror?an
(2016).Approachestoanalysisinmodel-basedcognitiveneuroscience.Journal
analysisofresponseprogramming.QuarterlyJournalofExperimentalPsychology,
ofMathematicalPsychology,.
29(4),727–743.http://doi.org/10.1080/14640747708400645.
Turner, B. M., Forstmann, B. U., Wagenmakers, E.-J., Brown, S. D., Seder-
Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85(2),59–108.
berg, P. B., & Steyvers, M. (2013). A Bayesian framework for simulta-
http://doi.org/10.1037/0033-295X.85.2.59.
neously modeling neural and behavioral data. NeuroImage, 72, 193–206.
Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: theory and http://doi.org/10.1016/j.neuroimage.2013.01.048.
data for two-choice decision tasks. Neural Computation, 20(4), 873–922. Turner,B.M.,vanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
http://doi.org/10.1162/neco.2008.12-06-420. stractionsthroughneuroimaging:theneuraldriftdiffusionmodel.Psychological
Ratcliff,R.,Philiastides,M.G.,&Sajda,P.(2009).Qualityofevidenceforperceptual Review,122(2),312–336.http://doi.org/10.1037/a0038894.
decisionmakingisindexedbytrial-to-trialvariabilityoftheEEG.Proceedings van Maanen, L., Brown, S. D., Eichele, T., Wagenmakers, E.-J., Ho, T., Ser-
oftheNationalAcademyofSciencesoftheUnitedStatesofAmerica,106(16), ences, J., et al. (2011). Neural correlates of trial-to-trial fluctuations
6539–6544.http://doi.org/10.1073/pnas.0812589106. in response caution. Journal of Neuroscience, 31(48), 17488–17495.
Ratcliff,R.,&Rouder,J.N.(1998).Modelingresponsetimesfortwo-choicedecisions. http://doi.org/10.1523/JNEUROSCI.2924-11.2011.
PsychologicalScience,9(5),347–356.http://doi.org/10.1111/1467-9280.00067. vanRavenzwaaij,D.,&Oberauer,K.(2009).Howtousethediffusionmodel:Param-
Rieskamp, J. (2008). The probabilistic nature of preferential choice. Journal of eterrecoveryofthreemethods:EZ,fast-dm,andDMAT.JournalofMathematical
ExperimentalPsychology:Learning,Memory,andCognition,34(6),1446–1465. Psychology,53(6),463–473.http://doi.org/10.1016/j.jmp.2009.09.004.
http://doi.org/10.1037/a0013646. vanVeen,V.,Krug,M.K.,&Carter,C.S.(2008).Theneuralandcomputationalbasis
Robinson,W.S.(1950).Ecologicalcorrelationsandthebehaviorofindividuals. ofcontrolledspeed-accuracytradeoffduringtaskperformance.JournalofCogni-
AmericanSociologicalReview,15(3),351–357. tiveNeuroscience,20(11),1952–1965.http://doi.org/10.1162/jocn.2008.20146.
Roitman, J. D., & Shadlen, M. N. (2002). Response of neurons in the lateral VonNeumann,J.,&Morgenstern,O.(1947).TheoryofGamesandEconomicBehavior
intraparietalareaduringacombinedvisualdiscriminationreactiontimetask. (2.Aufl.).Princeton,N.J.;Woodstock:PrincetonUniversityPress.
JournalofNeuroscience,22(21),9475–9489. Wiecki,T.V.,Sofer,I.,&Frank,M.J.(2013).HDDM:HierarchicalBayesianestimation
Shenhav,A.,Botvinick,M.M.,&Cohen,J.D.(2013).Theexpectedvalueofcontrol:an of the drift-diffusion model in python. Frontiers in Neuroinformatics, 7, 14.
integrativetheoryofanteriorcingulatecortexfunction.Neuron,79(2),217–240. http://doi.org/10.3389/fninf.2013.00014.
http://doi.org/10.1016/j.neuron.2013.07.007. Zandbelt, B., Purcell, B. A., Palmeri, T. J., Logan, G. D., & Schall, J. D. (2014).
Simen, P. (2012). Evidence accumulator or decision Threshold—Which cor- Responsetimesfromensemblesofaccumulators.ProceedingsoftheNational
tical mechanism are we observing? Frontiers in Psychology, 3, 183. Academy of Sciences of the United States of America, 111(7), 2848–2853.
http://doi.org/10.3389/fpsyg.2012.00183. http://doi.org/10.1073/pnas.1310577111.