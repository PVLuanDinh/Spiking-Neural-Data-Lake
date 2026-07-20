JournalofMathematicalPsychology76(2017)156–171
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Relating accumulator model parameters and neural dynamics
BradenA.Purcella,∗ ,ThomasJ.Palmerib
aCenterforNeuralScience,NewYorkUniversity,NewYork,NY10003,UnitedStates
bVanderbiltUniversity,UnitedStates
h i g h l i g h t s
• Neuralanalysesareusedtoquantifychangesinaccumulatormodeldynamics.
• Accumulatormodeldynamicsdistinguishmodelsthatbehavioralonecannot.
• However,analysisofdynamicsalonecannotpinpointunderlyingmodelparameters.
• Jointconsiderationofbehaviorandneuraldynamicsprovidesmaximalconstraint.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Accumulatormodelsexplaindecision-makingasanaccumulationofevidencetoaresponsethreshold.
Availableonline1August2016 Specific model parameters are associated with specific model mechanisms, such as the time when
accumulationbegins,theaveragerateofevidenceaccumulation,andthethreshold.Thesemechanisms
Keywords:
determineboththewithin-trialdynamicsofevidenceaccumulationandthepredictedbehavior.Cognitive
Decisionmaking
modelersusuallyinferwhatmechanismsvaryduringdecision-makingbyseeingwhatparametersvary
Accumulatormodels
whenamodelisfittedtoobservedbehavior.Therecentidentificationofneuralactivitywithevidence
Responsetime
Neuralmodels accumulationsuggeststhatitmaybepossibletodirectlyinferwhatmechanismsvaryfromananalysis
ofhowneuraldynamicsvary.However,evidenceaccumulationisoftennoisy,andnoisecomplicatesthe
relationshipbetweenaccumulatordynamicsandtheunderlyingmechanismsleadingtothosedynamics.
To understand what kinds of inferences can be made about decision-making mechanisms based on
measuresofneuraldynamics,wemeasuredsimulatedaccumulatormodeldynamicswhilesystematically
varyingmodelparameters.Insomecases,decision-makingmechanismscanbedirectlyinferredfrom
dynamics,allowingustodistinguishbetweenmodelsthatmakeidenticalbehavioralpredictions.Inother
cases, however, different parameterized mechanisms produce surprisingly similar dynamics, limiting
theinferencesthatcanbemadebasedonmeasuringdynamicsalone.Analyzingneuraldynamicscan
provideapowerfultooltoresolvemodelmimicryatthebehaviorallevel,butwecautionagainstdrawing
inferences based solely on neural analyses. Instead, simultaneous modeling of behavior and neural
dynamicsprovidesthemostpowerfulapproachtounderstanddecision-makingandlikelyotheraspects
ofcognitionandperception.
©2016ElsevierInc.Allrightsreserved.
Cognitivemodelingallowsustoinferthemechanismsunder- 1975; Nosofsky & Palmeri, 1997; Palmer, Huk, & Shadlen, 2005;
lyingperception,action,andcognitionbasedonobservedbehav- Ratcliff&McKoon,2008;Ratcliff&Smith,2004;Reddi&Carpen-
ior(Busemeyer&Diederich,2009;Farrell&Lewandowsky,2010; ter,2000;Shadlen,Hanks,Churchland,Kiani,&Yang,2006;Smith
Townsend&Ashby,1983).Inthedomainofdecision-making,ac- &Vickers,1988;Usher&McClelland,2001;Vickers,1979).These
cumulatormodels(alsocalledsequential-samplingmodels)provide modelsassumethatevidenceforaparticularresponseisintegrated
themostcompleteaccountofbehaviorformanydifferenttypesof over time by one or more accumulators. A response is selected
decisions(Bogacz,Brown,Moehlis,Holmes,&Cohen,2006;Brown when evidence reaches a response threshold. Variability in the
&Heathcote,2005,2008;Laming,1968;Link,1992;Link&Heath, timeittakesforaccumulatedevidencetoreachthresholdaccounts
forvariabilityinchoiceprobabilitiesandresponsetimesobserved
inabroadrangeofdecision-makingtasks.
∗ Correspondingauthor. Particular accumulator model parameters represent distinct
( )
E-mailaddress:braden@nyu.edu(B.A.Purcell). decision-making mechanisms (Fig. 1). An encoding time t e
http://dx.doi.org/10.1016/j.jmp.2016.07.001
0022-2496/©2016ElsevierInc.Allrightsreserved.

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 157
parameterdefinesthetimeforsensoryandperceptualprocessing,
| a drift rate                         | parameter |     | (v) defines | the | mean                       | rate of | evidence |     |     |     |     |     |
| ------------------------------------ | --------- | --- | ----------- | --- | -------------------------- | ------- | -------- | --- | --- | --- | --- | --- |
| accumulation,astartingpointparameter |           |     |             |     | ( z ) determinestheinitial |         |          |     |     |     |     |     |
( )
| state of   | an accumulator, |                | a threshold |         | parameter      | a          | defines the |     |     |     |     |     |
| ---------- | --------------- | -------------- | ----------- | ------- | -------------- | ---------- | ----------- | --- | --- | --- | --- | --- |
| level of   | evidence        | that           | must be     | reached | before         | a response | is          |     |     |     |     |     |
|            |                 |                |             | (       | )              |            |             |     |     |     |     |     |
| initiated, | and a           | motor response |             | time    | t m parameter  |            | defines the |     |     |     |     |     |
| time to    | execute         | a response     | (Fig.       | 1(A)).  | By identifying |            | parameter   |     |     |     |     |     |
valuesthatmaximizethematchbetweenobservedandpredicted
behavior(e.g.Vandekerckhove&Tuerlinckx,2007),themodelscan
| reveal the    | mechanisms |             | underlying   | variation     |              | in decision-making |            |     |     |     |     |     |
| ------------- | ---------- | ----------- | ------------ | ------------- | ------------ | ------------------ | ---------- | --- | --- | --- | --- | --- |
| behavior      | across     | different   | experimental |               | conditions.  | For                | example,   |     |     |     |     |     |
| manipulations | of         | speed       | versus       | accuracy      | instructions |                    | affect the |     |     |     |     |     |
| response      | threshold  | (Brown      | &            | Heathcote,    | 2008;        | Wagenmakers,       |            |     |     |     |     |     |
| Ratcliff,     | Gomez,     | & McKoon,   | 2008),       | manipulations |              | of                 | experience |     |     |     |     |     |
| (Nosofsky     | & Palmeri, | 1997;       | Palmeri,     |               | 1997;        | Petrov,            | Van Horn,  |     |     |     |     |     |
| & Ratcliff,   | 2011;      | Ratcliff,   | Thapar,      | & McKoon,     |              | 2006) or           | stimulus   |     |     |     |     |     |
| strength      | (Palmer    | et al.,     | 2005;        | Ratcliff      | & McKoon,    | 2008;              | Ratcliff   |     |     |     |     |     |
| & Rouder,     | 1998)      | affect      | drift rate,  | and           | manipulation |                    | of dynamic |     |     |     |     |     |
| stimulus      | noise      | prolong     | encoding     | time          | (Ratcliff    | & Smith,           | 2010).     |     |     |     |     |     |
| In addition,  | many       | accumulator |              | models        | assume       | that               | some of    |     |     |     |     |     |
thesemechanismscanvaryovertrialstoexplainwithin-condition
| variability | in behavior, |     | with additional |     | parameters | defining | the |     |     |     |     |     |
| ----------- | ------------ | --- | --------------- | --- | ---------- | -------- | --- | --- | --- | --- | --- | --- |
degreeofvariabilityinotherparameters.
| Recent        | neurophysiological |             |              | and neuroimaging |          | studies  | have       |     |     |     |     |     |
| ------------- | ------------------ | ----------- | ------------ | ---------------- | -------- | -------- | ---------- | --- | --- | --- | --- | --- |
| identified    | potential          | linking     | propositions |                  | (Schall, | 2004;    | Teller,    |     |     |     |     |     |
| 1984) between |                    | accumulator |              | models           | and      | measures | of brain   |     |     |     |     |     |
| activity      | (Forstmann,        | Ratcliff,   | &            | Wagenmakers,     |          | 2015;    | Forstmann, |     |     |     |     |     |
Wagenmakers,Eichele,Brown,&Serences,2011;Gold&Shadlen,
2007;Palmeri,Schall,&Logan,2014;Shadlen&Kiani,2013;Smith
&Ratcliff,2004).Differentapproacheshaveestablisheddifferent
| kinds of | connections |       | between  | models | and         | neural | measures   |     |     |     |     |     |
| -------- | ----------- | ----- | -------- | ------ | ----------- | ------ | ---------- | --- | --- | --- | --- | --- |
| (Turner, | Forstmann,  | Love, | Palmeri, | &      | Van Maanen, |        | 2017). One |     |     |     |     |     |
approachhasbeentofitamodeltobehavior,andusethefitted
parametersasatoolforinterpretingoridentifyingneuralsignals.
Correlatingmodelparametersandneuralsignalsacrosssubjects
Fig.1. Expectedrelationshipsbetweenaccumulatormodelparameters,modeldy-
namics,andneuraldynamics.A:Illustrationofaccumulatormodelparameters.Four
andconditionscanprovideinsightintowhatbrainregionsmight
primaryparametersdeterminethedecision-makingmechanisms:encodingtime
| be involved | in  | determining | the | model | threshold, | drift | rate, and | (t )            |                          |                              |                            |                                |
| ----------- | --- | ----------- | --- | ----- | ---------- | ----- | --------- | --------------- | ------------------------ | ---------------------------- | -------------------------- | ------------------------------ |
|             |     |             |     |       |            |       |           | e d e f i n e s | th e t im e f o r pe r c | ep t u a l p r o c e s sin g | p r e c e d i n g e vi d e | n c e a c c u m u l at i o n , |
non-decision time (Forstmann et al., 2008; Heekeren, Marrett, ( v ) ( )
|     |     |     |     |     |     |     |     | d r i ft r a t e | d ef in es t h e m e a n | r a t e o f a c c u m u la | ti o n , s t a r tin g p o i | n t z d e te r m i n e s |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------------------------ | -------------------------- | ---------------------------- | ------------------------ |
Bandettini,&Ungerleider,2004;vanMaanenetal.,2011;White, theinitialstate,threshold(a)definesthelevelofevidencethatmustbereached
|     |     |     |     |     |     |     |     | beforearesponseisinitiated,andmotorresponsetime(tm |     |     | )definesthetimetoexe- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --------------------- | --- |
Mumford,&Poldrack,2012).Anotherapproachhasbeentojointly
modelbehavioraldataandneuralresponsestogether,significantly cutearesponse.Fourcorrespondingstochasticparameters(swithsubscript)define
theacross-trialvariabilityforeachparameter(seeSection1).Inthesesimulations,
| constraining | parameter |     | estimates | (Cassey, |     | Gaunt, Steyvers, | &   |     |     |     |     |     |
| ------------ | --------- | --- | --------- | -------- | --- | ---------------- | --- | --- | --- | --- | --- | --- |
motortimewasalwaysassumedtobezero.B:Examplemodeldynamicsforfive
Brown,2016;Turner,vanMaanen,&Forstmann,2015).
simulatedtrialsusingidenticalparameters.Duringencodingtime,themodelactiv-
| Another | line | of work | suggests | that | the firing | rates | of certain |     |     |     |     |     |
| ------- | ---- | ------- | -------- | ---- | ---------- | ----- | ---------- | --- | --- | --- | --- | --- |
ityisfixedatthestartingpoint.Followingencodingtime,evidenceissampledfrom
neuralpopulationsdirectlyrepresenttheevidenceaccumulation adistributionwithmeanvandstandarddeviations(within-trialnoise,inset)and
processproposedintheaccumulatormodelframework.Inthese accumulatedovertime.Responsetimes(RTs,arrows)arethesumofencodingtime,
thetimeneededforaccumulatedevidencetoreachthreshold(i.e.,decisiontime).
studies, animals are trained to perform perceptual decision- Duetowithin-trialnoise,eventhesamesetofparametersproducesvariabilityin
making tasks and neural activity is recorded from one or more bothRTandtheevidenceaccumulationtrajectory.C:Examplesimulatedsingle-unit
intracranial electrodes simultaneously while animals perform activityandmeasuresofneuraldynamics.Toppanelsshowthatneuralactivityon
the task. Neural responses can then be analyzed aligned to the individualtrialsgivenbythespikedischargetimes(blackdots)alignedonstim-
ulusonset(left)orRT(right;redcircles).Individualspiketrainsarehighlynoisy,
| timing of | task | events (e.g., | stimulus |     | onset) | or the behavior | of  |     |     |     |     |     |
| --------- | ---- | ------------- | -------- | --- | ------ | --------------- | --- | --- | --- | --- | --- | --- |
buttheaveragefiringrateovertrialsrevealsunderlyingstructureinthedynamics
theanimal(e.g.,responseinitiation).Specifically,thefiringrates
(graylines,bottom).Fourmeasuresofneuraldynamicsarecommonlyappliedto
of neurons within a distributed network of areas including pre- makeinferencesaboutmodelparameters.Theonsetishypothesizedtocorrespond
frontal cortex (Ding & Gold, 2012; Hanes & Schall, 1996; Heitz totheencodingtime,thegrowthrateishypothesizedtocorrespondtothedriftrate,
& Schall, 2012; Kiani, Cueva, Reppas, & Newsome, 2014; Kim & thebaselineishypothesizedtocorrespondtothestartingpoint,andtheactivityat
Shadlen,1999;Mante,Sussillo,Shenoy,&Newsome,2013;Purcell RTishypothesizedtocorrespondtothethreshold.Dashedblacklinesillustratethe
computationofgrowthratebasedontheslopeofthelineconnectingtheactivity
et al., 2010; Purcell, Schall, Logan, & Palmeri, 2012), superior atonsettoactivityatRT.Neuralspiketimesweresimulatedaccordingtoatime
colliculus(Horwitz&Newsome,1999;Ratcliff,Cherian,&Segraves, inhomogeneousPoissonprocesswitharateparameterdeterminedbysimulated
accumulatormodeldynamics.
| 2003; Ratcliff, | Hasegawa, |     | Hasegawa, |     | Smith, | & Segraves, | 2007), |     |     |     |     |     |
| --------------- | --------- | --- | --------- | --- | ------ | ----------- | ------ | --- | --- | --- | --- | --- |
posteriorparietalcortex(Churchland,Kiani,&Shadlen,2008;de
Lafuente,Jazayeri,&Shadlen,2015;Mazurek,Roitman,Ditterich,& timedependingontheanimal’supcomingchoice.Consistentwith
Shadlen,2003;Roitman&Shadlen,2002),premotorcortex(Cisek, expected accumulator model dynamics, the rate of rise depends
2006;Thura&Cisek,2014;Thura,Cos,Trung,&Cisek,2014),and on stimulus strength and RT. Importantly, activity converges
basal ganglia (Ding & Gold, 2010) exhibit dynamics consistent to a fixed firing rate shortly before the response is initiated
withaccumulationofperceptualevidence.Followingtheonsetof regardless of the stimulus and RT, consistent with a threshold
a stimulus, the firing rates of these neurons gradually rise over mechanismfordecisiontermination(Hanes&Schall,1996).The

158 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
finding that certain neural populations might be implementing evidence accumulation (e.g., the diffusion model, Ratcliff, 1978;
thecomputationsproposedbyaccumulatormodelssuggeststhat Ratcliff & Rouder, 1998; linear ballistic accumulator (LBA), Brown
neuralactivityfromthesepopulationscanprovideawindowonto & Heathcote, 2008; leaky-competing accumulator (LCA), Usher &
thedynamicsofevidenceaccumulationinthebrain.
|     |     |     |     |     |     |     |     | McClelland, | 2001; linear | approach | to threshold |     | with ergodic | rate |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------ | -------- | ------------ | --- | ------------ | ---- |
Theproposedlinkbetweenaccumulatormodeldynamicsand
(LATER),Reddi&Carpenter,2000).Ourgoalherewasnottoevalu-
| neural dynamics |     | suggests | that | we could | infer | the parameters |     |     |     |     |     |     |     |     |
| --------------- | --- | -------- | ---- | -------- | ----- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ateaspecificarchitectureperse,buttoevaluatethedynamicspro-
| of the decision-making |     | process |     | directly | from | analyses | of neural |     |     |     |     |     |     |     |
| ---------------------- | --- | ------- | --- | -------- | ---- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- |
ducedbydifferentparameterizedmodelmechanismsthatcanbe
| dynamics. | Conversely, | the | best-fitting |     | parameters | of model | fits |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ------------ | --- | ---------- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
sharedacrossarchitectures.Therefore,weadoptedageneralmod-
tobehavioraldatacouldbeusedtodirectlygeneratepredictions elingframeworkthatincludedabroadrangeofdecision-making
aboutexpectedneuraldynamics.Forexample,onestraightforward mechanisms.Wefirstprovideabroadoverviewoftheframework
mapping for accumulator models associates the encoding time followedbyaformaldescriptionofthemodel.
parameterwiththemeasuredonsetofaccumulationwhenneural For biological plausibility, we modeled the decision process
activity first begins to rise from baseline, the starting point as accumulation of perceptual evidence to a positive response
parameter with measured baseline activity before the stimulus threshold (Fig. 1(B)). For our purposes, the model activation can
| turns on  | and before | accumulation |        | begins | to               | rise, the | drift rate |               |                 |       |            |      |          |       |
| --------- | ---------- | ------------ | ------ | ------ | ---------------- | --------- | ---------- | ------------- | --------------- | ----- | ---------- | ---- | -------- | ----- |
|           |            |              |        |        |                  |           |            | be identified | with the firing | rates | of neurons | that | increase | their |
| parameter | with       | measured     | growth | rate   | of accumulation, |           | and the    |               |                 |       |            |      |          |       |
activityinsupportofaparticularresponse(Mazureketal.,2003;
responsethresholdparameterwithmeasuredactivitypriortothe
Purcelletal.,2010;Ratcliffetal.,2007).Notethatthisapproach
| onset of | the observed | motor | response |     | (Fig. 1(C)). | If we | observe |     |     |     |     |     |     |     |
| -------- | ------------ | ----- | -------- | --- | ------------ | ----- | ------- | --- | --- | --- | --- | --- | --- | --- |
iscompatiblewithaccumulatormodels,likethediffusionmodel
| changes | in the | measured | neural | onset, | then without | ever | fitting |            |                       |     |              |     |         |         |
| ------- | ------ | -------- | ------ | ------ | ------------ | ---- | ------- | ---------- | --------------------- | --- | ------------ | --- | ------- | ------- |
|         |        |          |        |        |              |      |         | (Ratcliff, | 1978), that represent |     | the decision |     | process | using a |
amodeltodata,wemightassertthatdifferencesintheencoding
singleaccumulatorwithpositiveandnegativeresponsethresholds
| time parameter |     | in the accumulator |     | model | explain | why | people |     |     |     |     |     |     |     |
| -------------- | --- | ------------------ | --- | ----- | ------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
becausethesemodelscanusuallybereformulatedusingmultiple
| or animals | are    | slow in      | one condition |                | and fast | in another. | We      |               |            |             |          |         |            |          |
| ---------- | ------ | ------------ | ------------- | -------------- | -------- | ----------- | ------- | ------------- | ---------- | ----------- | -------- | ------- | ---------- | -------- |
|            |        |              |               |                |          |             |         | accumulators  | (Bogacz et | al., 2006;  | Ratcliff | et      | al., 2007; | Usher    |
| could also | choose | to constrain |               | an accumulator |          | model by    | setting |               |            |             |          |         |            |          |
|            |        |              |               |                |          |             |         | & McClelland, | 2001). For | simplicity, | we       | modeled | only       | a single |
theencodingtimeparameterequaltothemeasuredneuralonset
|             |             |         |     |        |       |            |         | accumulator, | but the framework |     | can be | naturally | extended | to  |
| ----------- | ----------- | ------- | --- | ------ | ----- | ---------- | ------- | ------------ | ----------------- | --- | ------ | --------- | -------- | --- |
| time across | conditions. | Working |     | in the | other | direction, | suppose |              |                   |     |        |           |          |     |
thatanaccumulatormodelfittedtobehaviorrequiredthatthere explainchoicebehaviorwiththeadditionofmoreaccumulators
be changes in the encoding time parameter across conditions. (e.g.,LBA,LATER,LCA).Tomaintainfocusonthedecisionmaking
This model could be tested by examining whether there are mechanisms and their resulting dynamics, we did not explore
concomitantchangesinthemeasuredneuralonset. modelsthatincludedcompetitionamongaccumulatorsorleakage
Of course, both of these approaches are valid only if the of accumulated information (Smith, 1995; Usher & McClelland,
mappingbetweenmodelparametersandneuraldynamicsisone- 2001); analyses of model dynamics may help to distinguish
to-one.Aone-to-onemappingbetweenparametersanddynamics betweenthesemechanismaswell(Bollimunta&Ditterich,2012;
is indicated when variation in a certain parameter corresponds Boucher,Palmeri,Logan,&Schall,2007;Purcelletal.,2010;Purcell,
tovariationinauniquemeasureofdynamics.Forexample,that Schalletal.,2012).
variationintheonsettimeisonlyassociatedwithvariationinthe Recallthatfourprimaryparametersdeterminehowaccumulator-
encodingtimeparameter(Fig.1(C)),andnotthedriftrate,starting model mechanisms can vary across experimental conditions
| point,orthreshold.Adirectmappingbetweenmodelparameters |     |      |        |               |     |         |         |                            |               | ( )     |                           |               |     |           |
| ------------------------------------------------------ | --- | ---- | ------ | ------------- | --- | ------- | ------- | -------------------------- | ------------- | ------- | ------------------------- | ------------- | --- | --------- |
|                                                        |     |      |        |               |     |         |         | (Fig.1(A)).Theencodingtime |               | T e     | determinesthedelaybetween |               |     |           |
| and dynamics                                           | may | seem | almost | self-evident, | but | several | factors |                            |               |         |                           |               |     |           |
|                                                        |     |      |        |               |     |         |         | the onset                  | of a stimulus | and the | start of                  | accumulation. |     | Neurally, |
canmakethingsmorecomplicatedthanintuitionmightsuggest.
thisparametercanbeassociatedwithafferentconductiondelays,
To begin with, noise is ubiquitous in neural activity (Faisal, including the time needed to encode a stimulus with respect to
Selen, & Wolpert, 2008) and many accumulator models assume potential responses. Typically, encoding time and post-decision
( )
that evidence is noisy (e.g. Ratcliff & Smith, 2004). As we will motortime T arecombinedintoasingle‘‘non-decisiontime’’
m
see,within-trialnoiseintroducesadditionalsourcesofvariability parameter(oftendenotedT orT )becausethetwoparameters
r er
suchthatqualitativelydifferentmodelparameterscansometimes areindistinguishableusingonlybehavior.However,becausethey
predictverysimilardynamics.Themappingbetweenaccumulator canbedistinguishedinmodelandneuraldynamics,wedistinguish
model parameters and neural dynamics can also be obscured betweenthemhere,focusingspecificallyonencodingtimebecause
because of averaging over trials that resulted in different RTs, themotortimeisrelativelyshortandinvariableforneurophysio-
whichisoftennecessarytocombatnoiseinanalysesofneuraldata
logicalsignalsidentifiedwithsomemotorresponseslikesaccadic
(Fig.1(C)).Ifnotdoneproperly,thisaveragingcanfurtherobscure
eyemovement(Scudder,Kaneko,&Fuchs,2002).Thedriftordrift
| themappingbetweenmodelparametersandneuraldynamics. |     |     |     |     |     |     |     | ( )    |                                                  |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------ | ------------------------------------------------ | --- | --- | --- | --- | --- |
|                                                    |     |     |     |     |     |     |     | rate V | determinesthemeanofperceptualevidenceforapartic- |     |     |     |     |     |
Tounderstandhowvariationinaccumulatormodelparameters
ularresponse,andthereforetherateofaccumulation.Thestarting
| relates | to variation | in  | neural | dynamics, | we  | quantified | how | ( )     |                                                    |     |     |     |     |     |
| ------- | ------------ | --- | ------ | --------- | --- | ---------- | --- | ------- | -------------------------------------------------- | --- | --- | --- | --- | --- |
|         |              |     |        |           |     |            |     | point Z | determinestheinitialstateofevidenceatthestartofthe |     |     |     |     |     |
variationinaccumulatormodelparametersrelatestovariationin ( )
|     |     |     |     |     |     |     |     | accumulationprocess.Thethreshold |     |     | A determinestheamountof |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | ----------------------- | --- | --- | --- |
modeldynamics(i.e.,thetrajectoryofaccumulatedevidence).To
evidencethatmustbeaccumulatedbeforeamotorresponseisini-
| do this, | we applied | commonly |     | used neurophysiological |     |     | analyses |     |     |     |     |     |     |     |
| -------- | ---------- | -------- | --- | ----------------------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
tiated,anddeterminesthetradeoffbetweenaccuracyandspeed.
| to characterize |     | model dynamics |     | generated | from | known | sets of |     |     |     |     |     |     |     |
| --------------- | --- | -------------- | --- | --------- | ---- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- |
Eachprimaryparameterisassociatedwithstochasticparameters
parameters.Inthefollowingsection,wedescribetheaccumulator
|     |     |     |     |     |     |     |     | that can | produce across-trial | variability |     | in behavior. |     | Following |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------------------- | ----------- | --- | ------------ | --- | --------- |
modelsthatweresimulated,theparametersofthosemodelsthat
were varied, the measures of accumulator dynamics that were common conventions (e.g. Brown & Heathcote, 2008; Ratcliff &
made, and how relationships between model parameters and Rouder,1998),weassumedthatencodingtimevariedaccordingto
dynamicswereassessed. auniformdistributionwithranges T ;driftratevariedaccordingto
|     |     |     |     |     |     |     |     | aGaussiandistributionwithstandarddeviations |     |     |     |     | V ;startingpoint |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | ---------------- | --- |
1. Generalsimulationmethods variedaccordingtoauniformdistributionwithranges ;threshold
Z
|     |     |     |     |     |     |     |     | varied according | to a uniform | distribution |     | with | range | s (some |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------------ | ------------ | --- | ---- | ----- | ------- |
A
1.1. Modeloverview previous models have assumed normally-distributed thresholds
|     |     |     |     |     |     |     |     | (e.g. Grice, | 1968, 1972), | but we | used a | uniform | distribution | to  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | ------ | ------ | ------- | ------------ | --- |
Researchers have proposed various accumulator model ar- eliminatethepossibilitythatthethresholdcouldbeplacedbelow
chitectures that make specific assumptions about the form of thestartingpoint).Inotherwords,primaryparametersdetermine

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 159
the mean of parameter distributions, and stochastic parameters distribution with mean V and standard deviation s , and ξ is a
V
determinethevariability.Weadopttheconventionofusingupper- within-trial Gaussian noise term with mean zero and standard
casevariablestorefertoprimaryparametersthatdefineafixed deviations.Theprocessterminateswhenmodelactivationexceeds
parameterdistributionovertrials,andlower-casevariablestorefer a threshold, a, which is sampled for that trial from a uniform
|     |     |     |     |     |     | distributionwithrange[A− |     |     | sA, | A+ sA]. |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | ------- | --- | --- | --- |
toasamplefromthedistributiononanindividualtrial.Tobeclear,
|     |     |     |     |     |     |     |     |     | 2   | 2   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
amodelassumingallfoursourcesofacross-trialvariabilitywould The predicted response time, RT, is the sum of the encoding
( )
benon-identifiablebasedonbehavioraldataalone,butvariability time t e ,thedecisiontime–thetimefortheactivationtoreach
inthefourmechanismscanpossiblybedistinguishablebasedon threshold – and the motor time (t ; here, set to zero). Because
m
analysisofaccumulatordynamics. firingratescannotdropbelowzero,weinitiallyexploredversions
Theothermajorsourceofvariabilityiswithin-trialnoise.Many of the model that included a lower reflecting bound at zero.
accumulator models assume that evidence is noisy (Fig. 1(B), However,wefoundthatourresultswerequalitativelysimilarwith
inset;Ratcliff,1978;Usher&McClelland,2001),perhapsbecause orwithoutsuchalowerbound,sowereportresultsfromamodel
of some combination of within-trial stimulus noise, momentary thatdoesnotincludealowerreflectingboundtobeclearthatthis
variationoftheevaluationofapercept,orintrinsicvariabilityinthe isnotacrucialfactorinfluencingourfindingsandconclusions.
brain;othermodelsdonotassumewithin-trialvariability(Brown By analyzing the relationship between model dynamics and
| & Heathcote, |     | 2005, 2008; | Reddi & | Carpenter, 2000). | Here, we |                   |     |     |          |        |           |               |     |
| ------------ | --- | ----------- | ------- | ----------------- | -------- | ----------------- | --- | --- | -------- | ------ | --------- | ------------- | --- |
|              |     |             |         |                   |          | model parameters, |     | we  | hoped to | inform | the types | of inferences |     |
exploreddifferentlevelsofwithin-trialnoisetounderstandwhat that can be made by analyzing real neural signals. We therefore
impact this noise might have on measures of model dynamics. adopted simulation and analysis methods designed to mimic
Following common convention (e.g. Link & Heath, 1975), we the type of data and analyses seen in actual neurophysiological
assumed that within-trial variability in drift rate is normally experiments. Although closed-form solutions are available for
distributedwithstandarddeviations(butseeJones&Dzhafarov, expectedbehavior,simulationsareneededtostudythedynamics
2014;Ratcliff,2013). at the level of individual simulated trials. We used Monte-Carlo
If we assume that the evidence in the accumulator maps simulations to generate expected single-trial model dynamics
onto the firing rate of a neuron, then the translation from rate (e.g.Tuerlinckx,Maris,Ratcliff,&DeBoeck,2001).Allsimulations
to spikes will introduce an additional source of noise in any usedasimulationtimestepdt /τ =0 . 1ms,andmodeltrajectories
measure of neural dynamics (Fig. 1(C); Churchland et al., 2011; weredownsampledto1msresolutiontomatchtheresolutionof
Nawrotetal.,2008;Smith,2010,2015).Thisadditionalsourceof
typicalneuralrecordings.
variabilitycanbeaddedtothemodeldynamicsbyassumingthat
|            |          |      |                   |                    |      | We simulated |             | different | parameter | sets      | to represent |                   | different |
| ---------- | -------- | ---- | ----------------- | ------------------ | ---- | ------------ | ----------- | --------- | --------- | --------- | ------------ | ----------------- | --------- |
| the neural | activity | is a | doubly-stochastic | process reflecting | both |              |             |           |           |           |              |                   |           |
|            |          |      |                   |                    |      | experimental | conditions. |           | For each  | parameter |              | set, we generated |           |
rate variability (trial-by-trial variation in accumulated evidence) 5000 simulated trials, with each trial providing one RT and
( )
and point-process variability (Poisson-like spiking noise). This one model activation trajectory, x t . In some cases, we chose
point-processvariabilityhasthepotentialtofurtherobscurethe parameter sets by systematically varying individual parameters
relationshipbetweenmodelparametersanddynamics.However, to understand the selective influence these parameters have on
because separate trials are generally recorded sequentially, the accumulator dynamics. In other cases, we randomly sampled
spikingnoiseisindependentacrosstrialsandcanbereducedby parametersfromspecifiedrangestoensurethatourresultswere
averaging. For the analyses reported in this paper, we reached not specific to a particular parameter set. For those simulations,
thesameconclusionsregardlessofwhetherwedirectlyanalyzed we sampled 2500 parameter sets from the following ranges:
the model trajectories or used the model trajectories to drive T : 0 . 15–0 . 5;V: 0 . 01–0 . 15;Z: 0 . 001–0 . 1;A: 0–0 . 15;s : 0–0 . 2;
|     |     |     |     |     |     | e   |     |     |     |     |     | Te  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
a Poisson process and then analyzed the resulting spike count . 1;s . 1;s .
|            |       |       |                 |                       |         | s V : 0–0 | Z : | 0–0 | A : 0–0 1. | These | parameters | generated | a   |
| ---------- | ----- | ----- | --------------- | --------------------- | ------- | --------- | --- | --- | ---------- | ----- | ---------- | --------- | --- |
| histograms | (Fig. | 1(C); | Purcell et al., | 2010; Purcell, Schall | et al., |           |     |     |            |       |            |           |     |
diversityofRTdistributionsthatspannedtherangeofRTsobserved
2012),atleastsolongasthenumberoftrialswas∼20ormore
|     |     |     |     |     |     | in typical | decision-making |     | experiments |     | with | humans | and non- |
| --- | --- | --- | --- | --- | --- | ---------- | --------------- | --- | ----------- | --- | ---- | ------ | -------- |
(most neurophysiological studies record many more). Here, we human primates. To ensure that each parameter set produced
reportdirectanalysesofthemodeltrajectoriestodemonstratethat realisticbehavioranddynamics,wealsoimposedseveralinclusion
<
ourconclusionsabouttherelationshipbetweenmodelparameters criteria. We required that Z A so that the model produced
anddynamicsarenotanartifactoftheconversionfromspikerates non-zerodecisiontimes.Inaddition,werequiredthattheleading
to spike times, nor are they a consequence of limited statistical edge(5thpercentile)ofthedecisiontimedistributionmustbeat
power.Inpractice,whencomparingneuralandmodeldynamics, least50mstoensureasufficientlylongtimeintervaloverwhich
oneshouldalwaystakespikingvariabilityintoaccounttoensure to measure accumulator dynamics. Finally, we required that at
that differences between model and neural dynamics are not least90%oftrialsreachedthethresholdwithin2.0sofdecision
explainedbydifferencesinstatisticalpower.
time(ourmaximumsimulationtime)topreventexcessivelylong
decisiontimesandreducecomputationaldemands.
1.2. Simulationdetails
|     |     |     |     |     |     | 1.3. Measuresofmodeldynamics |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- |
( )
| Atthestartofasimulatedtrial,modelactivation,x |     |     |     |     | t ,isfixed |     |     |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
atstartingpoint,z,whichissampledforthattrialfromauniform We applied several measures of model dynamics to identify
sZ,
distributionwithrange[Z − Z + sZ].Activationremainsatz the parameters underlying variability in RT. We considered
|                             |     |     | 2                           | 2   |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| throughouttheencodingtime,t |     |     | ,whichissampledforthattrial |     |     |     |     |     |     |     |     |     |     |
e two types of RT variability. First, we asked whether we could
| fromauniformdistributionwithrange[T |     |     |     | − s T, T + | s T].After |     |     |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | --- | ---------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
e 2 e 2 identify the parameters underlying within-condition variability
theencodingtime,modeldynamicsaregovernedbythefollowing
|     |     |     |     |     |     | in RT (i.e., | random, | across-trial |     | variability). | Here, | the | goal is to |
| --- | --- | --- | --- | --- | --- | ------------ | ------- | ------------ | --- | ------------- | ----- | --- | ---------- |
stochasticdifferentialequation(seeBogaczetal.,2006;Usher&
|     |     |     |     |     |     | infer which | stochastic |     | parameters | are | greater | than zero | given |
| --- | --- | --- | --- | --- | --- | ----------- | ---------- | --- | ---------- | --- | ------- | --------- | ----- |
McClelland,2001): the observed model trajectories. Second, we asked whether we
|     |    |     |     |     |     | could use | model | dynamics | to  | identify | mechanisms | underlying |     |
| --- | --- | --- | --- | --- | --- | --------- | ----- | -------- | --- | -------- | ---------- | ---------- | --- |
| dt  |     | dt  |     |     |     |           |       |          |     |          |            |            |     |
dx= v+ ξ across-conditionvariabilityinRT—forexample,variationduetoan
| τ   |     | τ   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
experimentalmanipulation.Here,eachsetofprimaryparameters
where x is the model activation (accumulated evidence), v is can be identified with a simulated experimental condition and
the drift rate, which is sampled for that trial from the Gaussian thegoalistoinferwhichprimaryparametershavechangedgiven

160 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
theobservedchangesinthemodeltrajectories.Inbothcases,the Fourth, we measured the growth rate of accumulation to
problemisnottrivialbecausewithin-trialnoisecontributestoall threshold. This is the average rate of accumulation from the
variabilityinRT. time when activity first began increasing (onset time) until the
Bothneuralandmodeltrajectoriesareverynoisyatthelevel
thresholdwasreached.Toquantifythegrowthrate,weestimated
of individual trials (Fig. 1), which makes it difficult to extract theslopeofmodelactivationfromtheonsettimeuntilRT.Thiswas
meaningful insights about model parameters based on single computedastheactivityatRTminusthebaseline,dividedbythe
trials alone. Signal averaging is the most common method of RTminustheonsettime(seeFig.1(C)).
| noise reduction | in analyses | of neural | data. Individual | trials are |     |     |     |     |     |     |     |     |
| --------------- | ----------- | --------- | ---------------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
first binned according to the condition or behavior of interest 2. Simulationresults
| and then | averaged in order | to eliminate | across-trial | noise and |     |     |     |     |     |     |     |     |
| -------- | ----------------- | ------------ | ------------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
revealcommonunderlyingstructure.Tomirrorthisapproachfor Variabilityinaccumulatormodelparameterscausesvariability
ourmodeldynamics,webinnedandaveragedmodeltrajectories
inpredictedbehaviorandmodeldynamics.Weaskedwhetherthe
beforeanalyzingthepatternofdynamics.Eachmeasureofmodel
sourceofvariabilitycouldbeinferrednotbyfittingparameterized
dynamics(describedbelow)wasappliedtotheaveragetrajectory
modelstobehavioraldata,asincognitivemodelingwork,butby
forindividualbins.
measuringaccumulatordynamicsdirectly,asinneurophysiolog-
Tounderstandsourcesofwithin-conditionvariabilityinRT,we
icalwork.Todothis,wesimulatedageneralaccumulatormodel
analyzedtheactivationtrajectoriesthatproduceddifferentRTsfor
architecturewhileintroducingvariabilityinparticularparameter-
asingleparameterset.Wedividedthe5000simulatedtrialsfor
|     |     |     |     |     | ized mechanisms. |     | We  | then measured | accumulator |     | dynamics | in  |
| --- | --- | --- | --- | --- | ---------------- | --- | --- | ------------- | ----------- | --- | -------- | --- |
eachconditionintobinsbasedonthepredictedRTandaveraged
|     |     |     |     |     | ways analogous |     | to how | neurons | are analyzed |     | in order | to deter- |
| --- | --- | --- | --- | --- | -------------- | --- | ------ | ------- | ------------ | --- | -------- | --------- |
trajectorieswithineachbin.Wechooseabinsizeof20because
minehowvariabilityinthosedynamicsrelatestothevariability
it was large enough to average out some across-trial noise, but weintroducedviamodelparameters.
smallenoughtolimittherangeofRTswithinasinglebin,andalso
approximatedthevaluesusedinphysiologicalstudies(e.g.Purcell
2.1. Within-conditionvariabilityinparametersanddynamics
etal.,2010;Purcell,Schalletal.,2012;Woodman,Kang,Thompson,
&Schall,2008).
|     |     |     |     |     | To understand |     | sources | of within-condition |     |     | variability | in RT, |
| --- | --- | --- | --- | --- | ------------- | --- | ------- | ------------------- | --- | --- | ----------- | ------ |
Tounderstandsourcesofacross-conditionvariabilityinRT,we
|     |     |     |     |     | we analyzed | the | model | trajectories | that | produced | different | RTs |
| --- | --- | --- | --- | --- | ----------- | --- | ----- | ------------ | ---- | -------- | --------- | --- |
evaluatedseveraldifferentapproachestobinningtrialsthathave
givenasetofparameters.Wefirstexaminedamodelarchitecture
beenemployedbypreviousneurophysiologicalstudies.First,one
|              |                  |        |                  |            | assuming     | no within-trial |           | noise    | (‘‘Noiseless’’, | s       | = 0) to    | validate |
| ------------ | ---------------- | ------ | ---------------- | ---------- | ------------ | --------------- | --------- | -------- | --------------- | ------- | ---------- | -------- |
| could simply | group all trials | from a | condition into a | single bin |              |                 |           |          |                 |         |            |          |
|              |                  |        |                  |            | our measures |                 | of neural | dynamics | and             | provide | a baseline | for      |
andaverage.Alternatively,onecouldanalyzeresponsedynamicsin
|     |     |     |     |     | comparison | to  | noisy | models. | We started | by  | simulating | four |
| --- | --- | --- | --- | --- | ---------- | --- | ----- | ------- | ---------- | --- | ---------- | ---- |
smallRTbinsasdescribedaboveandthenaveragingtheresulting
measurements over bins. Ultimately, we will show that both of versionsofthemodelassumingacross-trialvariabilityinonlyone
these methods are problematic for inferring model parameters of the four primary parameterized mechanisms (encoding time,
fromdynamics,butfocusingonthesubsetoftrialsthatresultin drift rate, starting point, threshold); in other words, in each of
moderately fast RTs (e.g., 30th–50th percentile) produces better the four model versions, only one of the stochastic parameters,
, ,
resultsforthesetofneuralmeasuredthatweevaluated. s s s , or s , was non-zero and all the rest were equal to
|     |     |     |     |     | Te V | Z   | A   |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
We adapted four measures of model dynamics that were zero. The left column of Fig. 2 illustrates model dynamics for
|     |     |     |     |     | each version | of  | the model | using | a representative |     | set of | primary |
| --- | --- | --- | --- | --- | ------------ | --- | --------- | ----- | ---------------- | --- | ------ | ------- |
developedtoanalyzeobservedneuralresponses(Fig.1(C);Pouget
|               |            |                     |           |            | parameter | values; | later, | we generalize |                 | beyond | this particular |      |
| ------------- | ---------- | ------------------- | --------- | ---------- | --------- | ------- | ------ | ------------- | --------------- | ------ | --------------- | ---- |
| et al., 2011; | Purcell et | al., 2010; Purcell, | Schall et | al., 2012; |           |         |        |               |                 |        |                 |      |
|               |            |                     |           |            | parameter | set.    | In the | absence       | of within-trial |        | noise, there    | is a |
Woodmanetal.,2008).Eachmeasurewasoriginallydefinedand
|     |     |     |     |     | clear one-to-one |     | mapping | between | variability |     | in manipulated |     |
| --- | --- | --- | --- | --- | ---------------- | --- | ------- | ------- | ----------- | --- | -------------- | --- |
developedtocorrespondconceptuallytoaparticularaccumulator
parametersandvariabilityinmeasureddynamics.Asanticipated
modelparameterintheabsenceofanysensorynoise.Wecanthen
|                                                    |     |     |     |     | by our earlier   | discussion, |             | variability | in encoding |     | time (     | T ) maps |
| -------------------------------------------------- | --- | --- | --- | --- | ---------------- | ----------- | ----------- | ----------- | ----------- | --- | ---------- | -------- |
| askhoweffectiveiseachmeasureoncenoiseisintroduced. |     |     |     |     |                  |             |             |             |             |     |            | e        |
|                                                    |     |     |     |     | onto variability |             | in measured | onset,      | variability |     | in drift ( | V ) maps |
First,wemeasuredtheonsettimeofaccumulationwhenactivity
|     |     |     |     |     |     |     |     |     |     |     |     | ( ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
firstincreasedabovethepre-stimulusbaselinelevels.Toquantify ontovariabilityinmeasuredrate,variabilityinstartingpoint Z
theonsettime,weusedasliding-windowalgorithm(+/−50ms) maps onto variability in measured baseline, and variability in
|     |     |     |     |     |     | ( ) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
that moves backward in 1 ms increments beginning at RT. The threshold A mapsontovariabilityinactivityatRT.
onsetofactivationwasdeterminedasthetimewhenthefollowing Toquantifythis,followingtypicalneuralanalyses,wegrouped
criteria are met: (1) activity no longer increases according to a trialsintobinsofsize20andplottedeachmeasureofmodeldy-
Spearmancorrelation(alpha= 0 . 05)withinthewindowaround namicsasafunctionofthemeanRTforallbins(Fig.3(A)).Apar-
thecurrenttime;(2)activityatthattimewaslowerthan50%of ticularmeasureofmodeldynamicscorrelatedwithRTonlyfora
themaximum;and(3),asthewindowwasmovedbackwardsin single stochastic parameter in the model. This is illustrated in
Fig.3(A)bythefactthatonlythediagonalpanelsshowanyrela-
time,thecorrelationremainsnonsignificantfor30ms.Ourresults
|     |     |     |     |     | tionship | between | manipulated |     | model parameters |     | and measured |     |
| --- | --- | --- | --- | --- | -------- | ------- | ----------- | --- | ---------------- | --- | ------------ | --- |
wereunchangedwhenweinsteadcomputedtheonsetasthetime
|     |     |     |     |     | dynamics.Withoutwithin-trialnoise |     |     |     | ( s | = 0 ) ,themeasureddy- |     |     |
| --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --------------------- | --- | --- |
atwhichactivationfirstexceeds10%ofthedistancefrombaseline
namicsofaccumulationandvariabilityintheresultingbehavior
tothreshold.
Second,wemeasuredthebaselineofaccumulation.Thisisthe areexclusivelydeterminedbyvariabilityinaparticularaccumula-
initiallevelofactivationimmediatelyfollowingstimulusonset.To tormodelparameter.Inthatcase,therelationshipbetweenmodel
quantifythebaseline,wecomputedtheaverageactivationinthe dynamicsandmodelparametersisonetoone,straightforwardand
| initial100msofeachsimulatedtrial,whichwasalwayslessthan |     |     |     |     | intuitive. |     |     |     |     |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
theminimumencodingtimeassumedbyourchoiceofparameters. The relationship between variability in accumulator model
Third, we measured the activity at RT. This is the level of parameters and measured model dynamics is more complicated
activation near the time when the response was initiated. To withtheadditionofwithin-trialnoise.TherightcolumnofFig.2
quantifytheactivityatRT,wefirstalignedallsimulatedtrialsto illustratesvariabilityinaccumulatordynamicsgeneratedwiththe
the time of the initiated response. The activity at RT was then samesetofparameters,butnowwiththeadditionofwithin-trial
computedastheaverageactivationinthe5mswindowcentered noise(Noisy,s = 0 . 1).Allfourparametersets,regardlessofthe
onresponsetime. underlyingsourceofvariability,producevariationinthemeasured

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 161
RTdistributions(seeSection1).Across2500sampledparameters
sets, Fig. 4 displays the average correlation between measured
dynamicandRTforeachofthemanipulatedstochasticparameters.
Asshowninthefigure,withoutanywithin-trialnoise(Noiseless,
s=0),thereisaclearone-to-onemappingbetweenthemeasures
ofmodeldynamicsandstochasticparameters.However,within-
trial noise of a standard magnitude (Noisy, s = 0 . 1) produces
strongcorrelationsbetweenthemeasuredonsetofactivityandRT
andbetweenthemeasuredgrowthrateofactivityandRT.
Fig.5providessomeintuitionaboutwhythedynamicsunder
casesofwithin-trialnoisetakethisform.Theleftpanelsillustrate
example trajectories sampled from different ranges of RTs (fast,
medium,andslow),whiletherightcolumnshowstheprobability
distributionofaccumulatedevidenceateachtimestepconditional
on the same RT range. Fast RTs result from positive samples
of noise that elevate the observed growth rate, even if the
underlyingmeanrateofgrowthisinvariable.SlowRTsoccurwhen
accumulated evidence meanders around the starting point, only
rising immediately prior to reaching the threshold. As a result,
theactualendofencodingtime(startofevidenceaccumulation)
mayhaveoccurredlongbeforeactivityfirstbeginsrisingtoward
threshold.
We found that the signal-to-noise ratio (SNR) of incoming
evidence is a key factor influencing the correlation between
measuredonsetandRT,andbetweenmeasuredgrowthrateand
RT. In the accumulator model framework, SNR is defined by the
( ) ( )
ratioofthemeandriftrate V overwithin-trialnoise s .Tostudy
theinfluenceofSNRonmodeldynamics,wesimulatedamodel
in which within-trial noise was theonly source of variability; in
other words, all other stochastic parameters were set equal to
zero.WevariedtheSNR,whileallotherprimaryparameterswere
randomlysampledasdescribedabove.Therelationshipbetween
onsetandRTishighlydependentonSNR.WhenSNRishigh
(≥10 )
,
Fig.2. Inferringsourcesofwithin-conditionvariabilityfrommodeldynamics.Each themodeltrajectoriesaremorelikelytorisedirectlytothreshold,
panelshowstheaveragemodeltrajectoriesdividedbyRTfordifferentsetsofpa- and so measured onset does not correlate with RT (Fig. 6(A)).
rameters.Foreachrow,onestochasticparameterwassettoapositivevalue,andthe <
remainingstochasticparametersweresettozero(row1—encodingtime,sTe =0.3; However,whentheSNRislow(roughly 5),theonsetcorrelates
row2—driftrate,sV =0.02;row3—startingpoint,sZ =0.05;row4—threshold, stronglywithRT.Inthiscase,thegrowthratecorrelateswithRT
sA = 0.04).Modeltrajectoriesshownintheleftcolumnincludednowithin-trial irrespectiveofSNRbecausethewithin-trialnoiseistheonlysource
noise(s = 0,‘‘Noiseless’’)andmodeltrajectoriesshownintherightcolumnin- ofvariabilityinthemodel.Incontrast,themeasuredbaselineand
cludedconventionallevelsofwithin-trialnoise(s = 0.1,‘‘Noisy’’).Forallpanels,
primaryparameterswerefixedatthefollowingvalues:Ter = 0.3,V = 0.1,Z = thresholdareunaffectedbySNR.
0.029,A=0.08.Followingstandardneurophysiologicalapproaches,modeltrajec-
toriesforeachsetofparametersweregroupedbyRTandaveragedinbinsofsize20 2.2. Across-conditionvariabilityinparametersanddynamics
(seeSection1).Fast,medium,andslowreferstotheaveragetrajectoryfortheRT
binsatthe25th,50th,and75thpercentile,respectively.Dashedlinesindicatethe
modelthresholdforeachRTgroup.Arrows,whenshown,highlightthetimewhen The analyses above focused on within-condition variability
thethresholdiscrossedforeachRTgroup. in model dynamics and stochastic parameters, but accumula-
tor models are also frequently used to infer changes in decision
onset and growth rate with RT. This was clearly evident in the making across experimental conditions (e.g., variation in stim-
correlationbetweenthemeasuredonsetandgrowthrateandRT ulus strength, speed–accuracy instructions, etc.). To understand
forallfourmodels.AsshowninFig.3(B),thereisarelationship whethermeasuringneuraldynamicscanbeusedtodirectlyiden-
betweenonsetandRT(firstrow)andbetweengrowthrateandRT tifychangesindecision-makingmechanismsacrossconditions,we
(secondrow)evenwhenaseeminglyincommensurateparameter alsotestedhowvariationinmeasuredaccumulatordynamicsisre-
wasvaried.Thismeansthatsimplyobservingchangesintheonset latedtomanipulatedvariationinprimarymodelparameters.
orgrowthrateofneuralsignalsasafunctionofRTisnotsufficient Quantifyingchangesinmodeldynamicsandneuraldynamics
to draw conclusions about parameterized sources of variability across experimental conditions raises a challenge of how to
in the decision process. Changes in the measured baseline and summarizedynamicsacrossmultipletrials.Forspeededdecision-
activityatRTcanbesomewhatdiagnostic,butchangesintheonset makingtasksinwhichasubjectdetermineswhentomaketheir
andgrowthratecouldwellreflectaccumulatornoise. response,thetotaltrialdurationwillvarysubstantiallyovertrials.
This‘‘one-to-many’’relationshipbetweenmeasureddynamics While it is straightforward to average RTs of different durations
and manipulated parameters was not unique to the particular since those are simply point observations, there is no clear way
set of parameters we used in the above illustrations, but was to average trajectories of different durations because those are
observedacrossabroadrangeofparametersets.Todemonstrate extendedtimeseries.Neurophysiologicalstudieshavedealtwith
this, we again simulated four model variants in which a single thisvariabilityindifferentways,butthechoiceofhowtoaverage
stochasticparameter(st,sv,sz,orsa)tookonavaluegreaterthan andbintrialscanhaveamarkedinfluenceontheformofresulting
zero. But instead of evaluating just a single set of parameters, measured dynamics that can help or hinder inferences about
we sampled from a broad range of values resulting in diverse underlyingparameters.

162 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
Fig.3. Correlationsbetweenaccumulatormodelparametersandmeasuresofmodeldynamicsformodelsimulationswithoutnoise(A,Noiseless)andwithconvention
levelsofnoise(B,Noisy).Foreachcolumn,onestochasticparameterwassettoapositivevalue,andtheremainingstochasticparametersweresettozero.Parametervalues
wereidenticaltothesimulationsshowninFig.2.Eachrowshowsonemeasureofmodeldynamics(onsettime,growthrate,baseline,activityatRT)appliedtotheaverage
trajectoriesofindividualRTbinsasafunctionoftheaverageRTforthatbin.Whenevidenceaccumulationisnoiseless,eachstochasticparameterproducescorrelations
betweenRTandasinglemeasureofneuraldynamics(aone-to-onemapping).Whenevidenceaccumulationisnoisy,multiplemeasuresofneuraldynamicscorrelatewith
RTforeachstochasticparameter(amany-to-onemapping).
Oneseeminglystraightforwardapproachistosimplyaverage inanRTandexitedtheprocess,howshoulditcontributetothe
acrossalltrialsregardlessoftheirRT.Acriticallimitationofthis average trajectory? Most accumulator models do not define the
approachisthatitrequiressomeexplicitassumptionabouthow stateofaccumulatedevidenceaftertheresponse(butseeMoran,
to treat post-response accumulator activity (Fig. 7(A)–(D)). For Teodorescu,&Usher,2015;Pleskac&Busemeyer,2010;Resulaj,
early time points, all of the model or neural trajectories can be Kiani,Wolpert,&Shadlen,2009).Followingtheresponse,activity
averaged together at every time step, but at progressively later might continue to accumulate above the threshold after the
time points, increasingly more of the trajectories will have hit decisionismade(Fig.7(A)),decaybacktobaseline(Fig.7(B)),or
thresholdandtriggeredaresponse.Onceatrajectoryhasresulted remainatthethreshold(Fig.7(C)).Alternatively,wemayexclude

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 163
|     |     |     |     | average | changes.      | As more  | trials  | drop     | out over | time,       | the | signal   |
| --- | --- | --- | --- | ------- | ------------- | -------- | ------- | -------- | -------- | ----------- | --- | -------- |
|     |     |     |     | becomes | progressively |          | noisier | as fewer | trials   | contribute  |     | to the   |
|     |     |     |     | average | and the       | dynamics | are     | dictated | by       | the sensory |     | noise of |
individualtrials.Asaresult,earlierandlatertimepointsmaybe
associatedwithverydifferentdynamics(e.g.,highergrowthrate
|     |     |     |     | early, lower | drift | rates | later), | resulting | in a | grand | average | that |
| --- | --- | --- | --- | ------------ | ----- | ----- | ------- | --------- | ---- | ----- | ------- | ---- |
isnonlinearandnon-representativeofindividualtrials(Fig.7(E),
greencurve).
|     |     |     |     | An alternative   |            | approach         | is              | to first  | group    | trials      | into           | small    |
| --- | --- | --- | --- | ---------------- | ---------- | ---------------- | --------------- | --------- | -------- | ----------- | -------------- | -------- |
|     |     |     |     | RT bins          | as in      | the preceding    | section,        |           | compute  | the         | measures       | of       |
|     |     |     |     | model dynamics   |            | for each         | RT              | bin, and  | then     | simply      | average        | the      |
|     |     |     |     | resulting        | measures   | over             | all bins        | (i.e.,    | imagine  | averaging   |                | all data |
|     |     |     |     | points in        | each       | panel            | of Fig.         | 3). This  | approach |             | will minimize  |          |
|     |     |     |     | the contribution |            | of post-response |                 | activity, |          | but it      | is susceptible |          |
|     |     |     |     | to the same      | influences |                  | of within-trial |           | noise    | highlighted |                | in the   |
previoussection.Fig.8illustratestheproblemforanexampleset
|     |     |     |     | of parameters  |     | in which  | across-condition |        | variability |       | is explained   |     |
| --- | --- | --- | --- | -------------- | --- | --------- | ---------------- | ------ | ----------- | ----- | -------------- | --- |
|     |     |     |     | solely through |     | different | values           | of the | drift       | rate, | all stochastic |     |
parameterswerefixedatzero,andtherewasaccumulatornoise
( = . )
s 0 1 .ForthefasterRTs,themeasuredonsetcloselymatches
|     |     |     |     | the actual    | end                                             | of encoding | time      | because   | evidence     |       | rises        | directly |
| --- | --- | --- | --- | ------------- | ----------------------------------------------- | ----------- | --------- | --------- | ------------ | ----- | ------------ | -------- |
|     |     |     |     | from baseline |                                                 | to the      | threshold | on        | those trials | (Fig. | 8(A)).       | For      |
|     |     |     |     | moderate      | and                                             | slow RTs,   | however,  | the       | measured     |       | onset        | is often |
|     |     |     |     | later than    | the                                             | actual end  | of        | encoding  | time         | (Fig. | 8(B) and     | (C)).    |
|     |     |     |     | BecauseV      | (signal)isdecreasingands(noise)remainsfixed,the |             |           |           |              |       |              |          |
|     |     |     |     | SNR drops     | across                                          | conditions, |           | resulting | in increased |       | correlations |          |
betweenonsetandRTeventhoughtheencodingtimeparameter
|     |     |     |     | remains    | fixed    | (Fig. 8(D)). | To  | summarize,   | under  |            | this approach, |     |
| --- | --- | --- | --- | ---------- | -------- | ------------ | --- | ------------ | ------ | ---------- | -------------- | --- |
|     |     |     |     | conditions | in which | the          | SNR | is different | across | conditions |                | may |
resultinnon-selectiveinfluencesonthemeasuredonset.
Onesimplesolutionistofocusonarelativelysmallrangeoffast
tomoderateRTs(e.g.,30thto50thpercentile).ThisrangeofRTsis
smallenoughtolimittheinfluenceofpost-responseactivityand
earlyenoughtoallowagoodestimateofthetrueencodingtime.
Fig.9showstheaveragemodeltrajectoriesforthe30th–50thRT
percentilegeneratedfromdifferentsetsofprimaryparameters;for
eachsetofsimulations,oneprimaryparameterwasvariedacross
conditions(encodingtime,startingpoint,driftrate,orthreshold),
whiletheotherprimaryparameterswereheldfixed,within-trial
Fig.4. AveragecorrelationsbetweenmeasuresofmodeldynamicsandRT.When
noisewasfixedat0.1,andstochasticparameterswerefixedatzero.
| evidence is | noiseless (black), correlations | between RT | and measures of model |     |     |     |     |     |     |     |     |     |
| ----------- | ------------------------------- | ---------- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dynamics(onset,growthrate,baseline,andactivityatRT)indicateasinglesourceof Ineachcase,themodeldynamicsqualitativelyreflecttheprimary
within-conditionvariability(stochasticparameters,x-axis).Whenevidenceisnoisy parameterthatwasvaried.Weappliedthisapproachtoabroad
(red),correlationsbetweenRTandmeasuredonsetandgrowthrateareobserved range of randomly sampled parameter sets that included within
regardlessofstochasticparameters.Foragivensetofsimulations,onestochastic .
|     |     |     |     | trial noise | (s = | 0 1; | see Methods), |     | and found | that | it captured |     |
| --- | --- | --- | --- | ----------- | ---- | ---- | ------------- | --- | --------- | ---- | ----------- | --- |
parameter(x-axis)andallprimaryparameterswererandomlysampledfroma
|     |     |     |     | variation | in primary | parameters |     | reasonably |     | well | even with | the |
| --- | --- | --- | --- | --------- | ---------- | ---------- | --- | ---------- | --- | ---- | --------- | --- |
rangeofvalues(seeSection1);theotherstochasticparametersweresettozero.
|     |     |     |     | inclusion | of within-trial |     | noise | (Fig. 10). | Other | solutions |     | such as |
| --- | --- | --- | --- | --------- | --------------- | --- | ----- | ---------- | ----- | --------- | --- | ------- |
Thesampledparametervalueswereusedtogenerate5000simulatedtrials.The
simulatedtrialsweredividedintoRTbinsofsize20,andthePearsoncorrelation examining the full distribution onset, growth rate, baseline, and
coefficientwascomputedbetweeneachmeasureofmodeldynamicsandthemean thresholdacrossallRTbinsordevelopingalternativemeasuresof
ofeachRTbin(seeFig.3).Thisprocesswasrepeated2500timesforeachstochastic neuraldynamicsmayworkaswell.Thecriticalpointisthatthelink
parameterandeachvalueofwithin-trialnoise.Panelsshowthemeanandstandard
deviationoftheresultingcorrelations.(Forinterpretationofthereferencestocolor betweenmodelparametersandneuraldynamicsisnotassumed,
inthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.) butisdirectlytestedthroughanalysisofmodeldynamics.
allpost-responseactivitiesfromtheanalysisofneuraltrajectories 3. Generaldiscussion
byclippingthetrajectoriesaftereachresponse(Fig.7(E)).
Unfortunately,wedonotknowwhichassumptionaboutpost- Thediscoverythatcertainneuralsignalsrepresentaccumulated
response activity is correct in actual neural data and different evidencehasraisedexcitingnewpossibilitiestoconstrainmodel
| assumptions | about post-response | activity | can produce very |            |      |          |     |        |          |     |         |      |
| ----------- | ------------------- | -------- | ---------------- | ---------- | ---- | -------- | --- | ------ | -------- | --- | ------- | ---- |
|             |                     |          |                  | parameters | with | measures | of  | neural | dynamics | and | predict | neu- |
differentdynamicswhenaveragedacrosstrials.Fig.7(E)illustrates ral dynamics based on parameters values fitted to behavior. To
thispointbyshowingthegrandaveragemodeltrajectoriesfrom understandtherelationshipbetweenaccumulatormodelparam-
thesamesetofparametersusedinFigs.2and3,butdifferingonly etersandneuraldynamics,weexaminedtherelationshipbetween
in the assumptions about post-response dynamics. The average accumulatormodelparametersandmodeldynamics.Wesystem-
trajectoryisdramaticallydifferentdependingonwhetheractivity aticallyvariedmodelparametervaluesoverlargerangesandquan-
continuestoaccumulate(red),decaystobaseline(blue)orremains tifiedmodeldynamicsbyadaptingestablishedneurophysiological
atthreshold(black).Simplydroppingpost-responseactivityand methods. Our simulations revealed both advantages and poten-
averaging across trials is also problematic for several reasons. tialpitfallsofdirectlyrelatingaccumulatormodelparametersto
As time progresses, the composition of RTs contributing to the measuredneuraldynamics.Ontheonehand,ourresultsprovide

164 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
Fig.5. Examplesingle-trialtrajectoriesandprobabilitydistributionofaccumulatedevidenceconditionalonRT.GroupingnoisymodeltrajectoriesbyRTproducesdelaysin
themeasuredonset.Allstochasticparametersweresettozeroandwithin-trialnoisewassetto0.1.(A–C)Examplesingle-trialtrajectories(coloredlines)andaverage(black
line)forsimulatedtrialsthatresultedinfast(A,0–0.2s),medium(B,0.3–0.4s),orslow(0.4–0.5s)RTs.(D–F)Probabilitydistributionforaccumulatedevidenceconditional
onRTsterminatingwiththespecifiedfast,medium,andslowranges.IndividualtrajectoriesandprobabilitiesweresmoothedwithaGaussianfilterforillustrationpurposes
only;nootheranalysisorfigureusedsmoothedtrajectories.(Forinterpretationofthereferencestocolorinthisfigurelegend,thereaderisreferredtothewebversionof
thisarticle.)
a simple example of using neural dynamics to distinguish alter- temptingtointerprettheseincreasesintheonsetactivityasdelays
nativemodelsthatmakeidenticalbehavioralpredictions.Onthe inthestartofevidenceaccumulation,butourresultsindicatethat
otherhand,ourresultsshowhowwithin-trialnoiseandtrialav- thismaysimplyreflectasignatureofnoisyevidenceaccumulation.
eraging can potentially obscure the relationship between model Theobservationthattheonsetofneuralactivitymayvarywith-
parameters and dynamics in misleading ways. In the following outexplicitvariationinencodingtimehasimportantimplications
discussion, we review advantages and challenges of analyzing forinferencesaboutthenatureofinformationflowacrossprocess-
neuraldynamicsasamodelselectiontool.Methodsthatattempt ingstages(Meyer,Osman,Irwin,&Yantis,1988;Miller,1982).Pre-
tosimultaneouslyexplainbothbehavioralandneuraldataprovide viousstudieshaveusedtheonsetofmotor-relatedneuralsignalsas
the strongest constraints on inferences about underlying mech- aproxyforthestartofadiscretestageofmotorprocessing(Miller,
anisms. We emphasize that, when possible, model and neural 1998;Miller,Ulrich,&Rinkenauer,1999;Osmanetal.,2000;Smul-
dynamicsshouldbedirectlycomparedtoavoidincorrectassump- ders,Kok,Kenemans,&Bashore,1995;Woodmanetal.,2008).Ex-
tionsaboutlinksbetweenmeasuresofneuraldynamicsandmodel perimentalmanipulationsthatprolongonsettimehavebeeninter-
parameters. pretedasdelaysinthestartofmotorprocessing.Oursimulations
showthatthemeasuredonsetofneuralresponsesmaybesubstan-
3.1. Potentialpitfallsofinferringparametersfrommodelandneural tiallydifferentfromtheactualstarttimeofevidenceaccumulation.
dynamics Evenacontinuousflowofinformationtotheaccumulatornetwork
canproducecorrelationsbetweentheonsetandRTorvariationof
The discoveries that link accumulator models and neural onsetwithstimulusstrength.
dynamics(Hanes&Schall,1996;Roitman&Shadlen,2002)might We found that the relationship between the encoding time
seem to suggest that we can bypass fitting models to behavior parameter and the measured onset depends critically on SNR. If
altogetherandsimply‘‘readout’’theunderlyingparametersfrom the SNR is low (roughly less than 5; see Fig. 6), then one can
the neural dynamics. Our simulations highlight some of the expectreasonablystrongcorrelationsbetweenonsetandRTeven
significantdangersofinferringparametersfromdynamicsinthe in the absence of any explicit encoding time variability. SNR is
absenceofbehavioralmodelfitting. tightlylinkedtostimulusstrength,whichisoftenmanipulatedin
Oursimulationsshowthattheonsetofneuralactivitydoesnot psychophysicalstudies(e.g.Palmeretal.,2005),meaningthatthe
necessarilyreflecttheendofencodingtime,acomponentofthe relationshipbetweenmodelparametersanddynamicsmaydiffer
T parameterinaccumulatormodels;introducingvariabilityinthe across conditions of the same experiment. Strong correlations
er
driftrate,startingpoint,andthresholdparametersallmanifestas between onset and RT may be observed for weaker, but not
variabilityinmeasuredonset.Thisresulthasimportantimplica- stronger stimuli, even with a fixed unvarying encoding time
tions for the interpretation of neural signals in decision-making (Fig.8).Intheabsenceofbehavioralmodelfits,changesintheonset
experiments.Duringavarietyofperceptualdecision-makingtasks, time alone are not conclusively diagnostic about the underlying
theonsettimewhenfiringrateincreasescorrelatesstronglywith causeofbehavioralvariability.
stimulusstrengthandRTinpre-frontalcortex(DiCarlo&Maunsell, Our simulations highlight how methods of binning and
2005;Pougetetal.,2011;Purcelletal.,2010;Purcell,Schalletal., averagingovertrialscandramaticallyinfluencetheresultingform
2012;Woodmanetal.,2008)andsuperiorcolliculus(Ratcliffetal., of the measured dynamics. Simply averaging trajectories for all
2003,2007).SimilarincreasesintheonsettimewithRTaresome- trials, as is common in many neural analyses, produces average
timesobservedinparietalneurons(Cook&Maunsell,2002).Itis trajectoriesthatarehighlydependentonassumptionsaboutthe

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 165
Fig.7. Differentassumptionsaboutpost-responseactivitycanstronglyinfluence
average model dynamics. (A–C) Four illustrations of potential post-response
dynamics on single trials. (A, red) Evidence accumulation continues after the
thresholdisreached,(B,blue)evidencedecaysbacktostartingpoint,(C,black)
evidenceremainsatthethreshold,and(D,green)evidenceisclippedwhenthe
Fig.6. CorrelationsbetweenRTandmeasuredonsetdependonthesignal-to-
noiseratio(SNR,v/s).Primaryparameterswererandomlysampledfromarangeof thresholdisreached.(E)Averagedynamicsover5000simulatedtrialsforthesame
setofmodelparameters,butassumingdifferentformsofpost-responsedynamics.
valuesandthelevelofwithin-trialnoisewaschosentosystematicallyvarytheSNR.
ParametervaluesareidenticaltothoseusedforFig.2.Colorsindicatepost-response
Stochasticparameterswerefixedatzero.Foreachparameterset,thecorrelation
dynamicsillustratedinA–D.Thegreenlineinwhichactivityonindividualtrialsis
betweenRTandmeasuresofmodeldynamicswascomputedasinFig.4.
clippedatRTbecomesnoisierastimeprogressesbecausefewertrialscontributeto
theaverage.(Forinterpretationofthereferencestocolorinthisfigurelegend,the
form of post-response activity. Unless the behavioral state of readerisreferredtothewebversionofthisarticle.)
the subject after the response is carefully controlled during an
experimentandthemodelmakesclearpredictionsabouttheform additionalinsightsabouttheunderlyingparameters.Forexample,
of post-response dynamics (e.g. Moran et al., 2015; Pleskac & although our analyses focused specifically on analyses of mean
Busemeyer, 2010; Resulaj et al., 2009), then it is risky to draw activity, the trial-to-trial variance and covariance of neural
strongconclusionsthatdependheavilyonuncheckedassumptions trajectories over time can reveal important insights about the
about post-response activity. Binning and averaging by small underlying parameters (Churchland et al., 2011). For example,
groupsofRTsbeforeapplyingmeasuresofdynamicsisoneway Churchlandetal.(2011)showedthatthetrial-to-trialvarianceof
to exclude post-response activity, but simply averaging over all LIPneuronsexhibitedsignaturesofnoisyevidenceaccumulation.
RT bins can produce misleading results because the relationship SimilarresponseshavebeenobservedinFEFmovementneurons
betweenthemeasuredonsetandRTchangeswithSNR(Figs.6(A) (Purcell,Heitz,Cohen,&Schall,2012).
and8).Onepossiblesolutionistoexaminechangesinmeasures
of model dynamics over the full distribution of RT bins; for 3.2. Neuraldynamicsasamodelselectiontool
example, as drift rate increases, how does the measured onset
changeforeverysingleRTbin?However,wefoundthatasimpler Our simulations demonstrate a simple illustration of how
method may suffice. Binning over a limited subset of trials that neural dynamics might be used to resolve behavioral model
producedfasttomoderateRTs(∼30thto50thpercentile)seemsto
mimicry.Inmanyaccumulatormodelframeworks,variationinthe
provideasimpleandreliablemethodfortrackingacross-condition starting point and variation in threshold make indistinguishable
parametervariability,atleastinthesubsetofmodelarchitectures predictionsaboutvariationinbehaviorbecausetheybothchange
andmodelparameterizationsthatweexploredhere. the total evidence that must be accumulated for a choice.
Of course, the analyses of model and neural dynamics that But these parameters have very different implications for the
we applied here are not exhaustive. We selected four measures underlying neural mechanisms. For example, adjustments in
(onset, growth rate, baseline, and activity at RT) to inform the neural activity at the time of RT are associated with changes in
interpretationofpastandfutureneurophysiologicalstudiesthat the strength of cortico-striatal connections (Lo & Wang, 2006),
usedsimilarapproaches(e.g.Woodmanetal.,2008).Alternative whereasadjustmentsofthebaselinemaybeimplementedthrough
methods for quantifying dynamics could potentially provide backgroundexcitationandinhibition(Lo,Wang,&Wang,2015).

166 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
Fig.8. Across-conditionvariationofdriftratecanproducevariationinmeasured
onset.(A–C)Simulatedmodeldynamicsforthreevaluesofdriftratedividedinto
fastRTs(A,25thpercentile),mediumRTs(B,50thpercentile),andslowRTs(C,75th
percentile).ForslowRTs,theonsetappearstoincreasewithdriftrate,althoughthe
encodingtimeparameterwasunchanged.(D)Eachmeasureofmodeldynamics
wascomputedforindividualRTbinsofsize20,andtheresultingmeasureswere
averagedoverallRTbinsforagivensetofparameters.Theaverageonsetdecreases
withdriftrateduetochangesintheSNR.
|     |     |     |     |     |     |     | Fig. 9. Averaging | model | trajectories | for the | 30th to | 50th percentile | of the |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | ----- | ------------ | ------- | ------- | --------------- | ------ |
Although the measured onset time and growth rate of neural RT distribution produces qualitative changes in dynamics that match model
dynamics appear to be less diagnostic about specific underlying parameters.Foreachsetofsimulationsoneprimaryparameterwasvaried(rows)
parameters, variation in measured neural baseline and neural whiletherestwereheldconstant.Allstochasticparameterswerefixedatzeroand
within-trialnoisewassetto0.1.
activityatRTcanbeassociatedmoredirectlywithvariationinthe
| starting point | and | threshold | parameters | of  | accumulator | models. |     |     |     |     |     |     |     |
| -------------- | --- | --------- | ---------- | --- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
evidenceaccumulationprovidedanexcellentaccountofbehavior,
| Examining | neural | dynamics | can | reveal underlying |     | mechanisms |     |     |     |     |     |     |     |
| --------- | ------ | -------- | --- | ----------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
butfailedtoexplaintheobservedneuraldynamics.Thefailureof
| that can | be invisible | based | on fits | of models | to behavioral |     | data |     |     |     |     |     |     |
| -------- | ------------ | ----- | ------- | --------- | ------------- | --- | ---- | --- | --- | --- | --- | --- | --- |
theleakymodelsmotivatedanovelgatingmechanisminwhich
alone.
|     |     |     |     |     |     |     | evidence | accumulation | begins | only after | the | evidence | exceeds |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | ------ | ---------- | --- | -------- | ------- |
Anumberofrecentstudieshaveprovidedfurtherillustrations
|                      |     |            |          |              |           |      | some constant | level.    | In addition | to providing   |     | an excellent | fit to |
| -------------------- | --- | ---------- | -------- | ------------ | --------- | ---- | ------------- | --------- | ----------- | -------------- | --- | ------------ | ------ |
| of the power         | of  | neural     | dynamics | as a model   | selection | tool | in            |           |             |                |     |              |        |
|                      |     |            |          |              |           |      | behavior,     | the gated | model       | also predicted | the | observed     | neural |
| diverse experimental |     | paradigms. |          | For example, | behavior  | in   | stop-         |           |             |                |     |              |        |
dynamics.
| signal tasks, | in  | which | subjects | must withhold | a   | preplanned |     |     |     |     |     |     |     |
| ------------- | --- | ----- | -------- | ------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Analysesofneuraldynamicshavealsorevealedcrucialinsights
responsefollowinganinfrequentcuetostop,hastraditionallybeen
aboutthemechanismsunderlyingstrategicadjustmentsofspeed
explainedasaracebetweenindependentprocessesrepresenting
andaccuracy.Traditionally,accumulatormodelsexplaintradeoffs
goingandstopping(Logan&Cowan,1984).Boucheretal.(2007)
implemented the stop and go processes as either independent between speed and accuracy through adjustments in the model
racingaccumulatorsorinteractiveracingaccumulatorsthatinhibit threshold (e.g. Brown & Heathcote, 2008; Palmer et al., 2005;
oneanother(Boucheretal.,2007).Bothmodelsexplainedbehavior Wagenmakers et al., 2008). Surprisingly, recent experiments in
equallywell,buttheinteractivemodelprovidedabetteraccount whichmonkeysweretrainedtoemphasizespeedoraccuracyin
ofneuralresponsesfromFEFmovementneurons.Morerecently, differentblocksdidnotshowtheexpectedchangesinthefiring
|     |     |     |     |     |     |     | rates of LIP | and FEF | neurons | around | the time | of the | response |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | ------- | ------ | -------- | ------ | -------- |
Logan,Yamaguchi,Schall,andPalmeri(2015)showedthatmodels
|                  |          |            |        |           |               |      | (Hanks, Kiani, | & Shadlen, |        | 2014; Heitz | & Schall, | 2012).   | Instead, |
| ---------------- | -------- | ---------- | ------ | --------- | ------------- | ---- | -------------- | ---------- | ------ | ----------- | --------- | -------- | -------- |
| in which         | stopping | inhibition | is not | direct,   | but instead   | acts | as a           |            |        |             |           |          |          |
|                  |          |            |        |           |               |      | neurons in     | both areas | showed | changes     | in the    | baseline | activity |
| gating mechanism |          | on the     | input  | to the go | unit, provide | an   | even           |            |        |             |           |          |          |
andinthedynamicsoftheevidenceaccumulationprocess;when
betteraccountoftheneuraldynamics(Loganetal.,2015).
As another example, in the domain of visual search, Purcell, speedwasemphasized,thegrowthrateofneuralresponseswas
Schall et al. (2012) and Purcell et al. (2010) evaluated different strongerirrespectiveofthesubject’schoice.Onewaytoreconcile
accumulator models based on their ability to explain not the model and the FEF data was to extend the model with a
only behavior (RT and accuracy), but also the dynamics of secondstageofaccumulation(Heitz&Schall,2012).Alternatively,
FEF movement neurons thought to implement the evidence the LIP data were explained by assuming that adjustments in
accumulationprocessforthistask.Modelsinwhichevidencewas threshold are implemented through an evidence-independent
not integrated at all or integrated perfectly could be ruled out urgency signal (Churchland et al., 2008; Hanks et al., 2014;
basedonpoorfitstobehavioralone.Modelsthatassumedleaky Thura&Cisek,2014).Increasingurgencycausesresponsestorise

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 167
Fig.10. Measuresofmodeldynamicscantrackchangesinprimaryparameterswhenmodeltrajectoriesareaveragedforthe30thto50thpercentileoftheRTdistributions.
Primaryparameterswererandomlysampledfromarangeofvalues(seeSection1),stochasticparameterswerefixedatzero,andwithin-trialnoisewasfixedat0.1.Panels
showthemeanandstandarddeviationforeachmeasureofmodeldynamics(rows)groupedintodecilesbyprimaryparametervalues(columns).Eachmeasureofmodel
dynamicscapturesvariationinoneprimaryparameterwhenonlyarestrictedrangeofRTsisaveraged.Notethatitwaspossibleforourinclusioncriteria(seeSection1)to
inducespuriouscorrelationsbetweenparametersandmeasureofmodeldynamics;forexample,becausethestartingpointmustbelowerthanthethreshold,increasing
thethresholdcouldproduceincreasedbaselinesimplybecausethepossiblerangeofstartingpointsincreased.Toeliminatethesespuriouscorrelations,weusedalimited
rangeofparametervaluesforstartingpointandthreshold.
faster irrespective of incoming evidence, leading to faster but
moreerror-proneresponses,whereasdecreasingurgencyallows
more time for evidence accumulation, leading to slower and
more accurate responses. Similar analyses of neural dynamics
suggest that urgency signals may implement adjustments of
response threshold following errors (Purcell & Kiani, 2016).
Recentbehavioralmodelingsuggeststhatthestrengthofurgency
mayvaryconsiderablyoversubjectsandexperiments(Hawkins,
Forstmann,Wagenmakers,Ratcliff,&Brown,2015),butthismodel
mechanismcurrentlyprovidesthemostparsimoniousexplanation
forbothbehavioralandneuraldata.
3.3. Modelingbehavioralandneuraldynamics
Fig.11. Illustrationofdifferentapproachestorelatingcognitivemodels,behavior,
andneuraldynamics.Weassumethatthebehaviorandneuraldynamicswere
While neural dynamics can provide a powerful tool to select
observed simultaneously in the context of some task, and neural dynamics
amongcompetingmodels,analysisofdynamicsalonecanproduce
may be analyzed in relation to task events or observed behavior. Arrow 1:
misleading inferences about the underlying mechanisms. We Traditional neurophysiological approaches relate neural dynamics directly to
recommend a modeling approach in which both behavioral and behavior(e.g.,howdoneuralresponseschangeundertaskconditionsinwhich
choicesaremoreorlessaccurate?).Arrow2:Traditionalcognitivemodelinginfers
neuraldataarejointlyconsidered;severalapproachestodoingso
modelparametersbasedonfitstobehavior(e.g.,howdomodelparameterschange
havebeenimplemented(Turneretal.,2017).
whenfittedtobehaviorfromconditionsassociatedwithhigherorloweraccuracy?).
Fig.11summarizessomedifferentapproachesthathavebeen Arrow 3: One approach to model-based cognitive neuroscience relates model
usedtounderstandbehavioralandneuraldata.Traditionalneuro- parametersdirectlytoobservedneuraldynamics(e.g.,howdoBOLDresponsesina
particularbrainregioncorrelatewithmodelparametersfittedtobehavior?).Arrow
physiologicalapproachesinvolvedirectlyanalyzingneuraldynam-
4:Analternativeapproachtomodel-basedcognitiveneuroscienceinvolvesadirect
icstoinferthemechanismsthatgiverisetobehavior(arrow1).For comparisonbetweenmodeldynamicsandneuraldynamics(e.g.,howwelldoesthe
example,correlationsbetweenneuralresponsesandbehaviorare modelreplicatebothbehavioralandneuraldynamicsacrossdifferentconditions?).
takenasevidencethattheneuralpopulationissomehowinvolved
ingeneratingthebehavior.Directmanipulationssuchaselectrical precisemechanismslinkingneuralresponsestoobservedbehav-
microstimulationandpharmacologicalinactivationcanhelpinfer ioroftenremainmurky.
acausalroleofaparticularbrainregioninthebehavior(Fetsch, Traditional cognitive modeling involves making inferences
Kiani,Newsome,&Shadlen,2014;Gold&Shadlen,2000;Hanks, aboutmechanismsbyoptimizingmodelparameterstomaximize
Ditterich,&Shadlen,2006;Monosov&Thompson,2009;Stuphorn the correspondence between predicted and observed behaviors
& Schall, 2006). However, without the aid of formal models, the (arrow 2 in Fig. 11). This approach makes assumptions about

168 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
the underlying cognitive mechanisms explicit and facilitates instantiated via the encoding time, starting point, and drift rate
rigorouscomparisonofalternativemodelsthroughevaluationof parameter. This neurally-constrained model provided a good fit
quantitativepredictions.Intheabsenceofneuraldata,however, to behavior, indicating that variability in the responses of the
these models make no explicit predictions for where and how visually-responsive neurons was sufficient to explain behavioral
these cognitive mechanisms are implemented in the brain. variabilityforthistask.Notethatthisapproachisdistinctfromthe
Additionally,asnotedabove,behavioralonemaybeinadequateto methoddescribedabovebecausethereisnocomparisonofneural
distinguishcompetingmodels. and model dynamics; instead, the observed neural dynamics
Model-based cognitive neuroscience aims to combine the replacetheparameterizedmodelmechanisms.
power of cognitive modeling with measures of neural activity. Complementing Purcell, Schall et al. (2012) and Purcell et al.
Oneapproachinvolvesdirectlyrelatingmodelparametervaluesto (2010), Cassey and colleagues (Cassey et al., 2016) recently
observedneuraldynamics(arrow3inFig.11).Aresearchermight developed a joint modeling approach that simultaneously fitted
firstfitamodeltobehavioraldataandthenidentifycorrelations
|         |              |            |        |     |          |        | the neural | dynamics | and | used         | the model | specification |     | of those  |
| ------- | ------------ | ---------- | ------ | --- | -------- | ------ | ---------- | -------- | --- | ------------ | --------- | ------------- | --- | --------- |
| between | fitted model | parameters | values | and | observed | neural |            |          |     |              |           |               |     |           |
|         |              |            |        |     |          |        | dynamics   | to drive | an  | accumulation |           | of evidence.  |     | With this |
measures(Mulder,vanMaanen,&Forstmann,2014).Forexample,
|     |     |     |     |     |     |     | approach, | the influence |     | of different | parameters |     | on the | expected |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------------- | --- | ------------ | ---------- | --- | ------ | -------- |
subject-to-subject variation in the magnitude of the response dynamicsisdirectlyincorporatedintothefittingprocess,obviating
threshold parameter is correlated with variation in the strength the need for proposing and validating connections between
ofBOLDresponsesinstriatumandpre-supplementarymotorarea specificmeasuresofdynamicsandmodelparameters.
(Forstmannetal.,2008).Thisapproachcanidentifyparticularbrain
areasthataresomehowrelatedtothemechanismscontrolledby
3.4. Linkingpropositions
modelparameters(O’Reilly&Mars,2011),providingausefultool
forinterpretingneuraldata.
Wehaveprimarilymadeconnectionstoworkusingsingle-unit
Apotentiallymorepowerfulapproachistosimultaneouslyfit
recordingsfromindividualneuronsinawakeandbehavingnon-
theneuralandbehavioraldatabyjointlymaximizingthelikelihood
of behavior and dynamics. For example, Turner and colleagues humanprimatesbecausetheseresponseshavebeenmoststrongly
|              |                    |                |           |         |              |              | identified         | with            | a representation |                    | of           | the evidence  | accumulation |              |
| ------------ | ------------------ | -------------- | --------- | ------- | ------------ | ------------ | ------------------ | --------------- | ---------------- | ------------------ | ------------ | ------------- | ------------ | ------------ |
| developed    | a method to        | simultaneously |           | fit     | model        | parameters   |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | process.           | To date,        | most             | neurophysiological |              | studies       |              | of this type |
| to behavior  | while also         | constraining   | parameter |         | estimates    | based        |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | are limited        | to              | recordings       | within             | a            | single region | of           | the brain,   |
| on patterns  | of BOLD activation |                | in cortex | (Turner | et           | al., 2015).  |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | raising the        | question        | of               | whether            | it is        | valid to      | draw         | conclusions  |
| An important | advantage          | of this        | approach  | is      | that it      | incorporates |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | about the          | decision-making |                  | process            |              | based         | on analyses  | of a         |
| constraints  | from the neural    | data           | as        | part of | the fitting  | process,     |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | limited population |                 | of cells.        | It                 | is highly    | doubtful      | that         | any one      |
| allowing     | one to identify    | whether        | a         | subset  | of parameter | space        |                    |                 |                  |                    |              |               |              |              |
|              |                    |                |           |         |              |              | region implements  |                 | the              | evidence           | accumulation |               | process      | alone.       |
canjointlyexplainboththebehavioralandneuraldata.Still,itis
common that the link that is established is one between model Rather, neurophysiological evidence suggests that the evidence
parametersandneuralmeasures(arrow3inFig.11). accumulation process is represented by a network of brain
Our simulations suggest, at least when appropriate forms of regions distributed throughout cortical and subcortical regions
neuraldataareavailable,thatthebestpracticeistoestablishlinks (Schall, 2013; Shadlen & Kiani, 2013). When different regions
|     |     |     |     |     |     |     | of the network |     | are observed |     | during | the same | decision | making |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | --- | ------ | -------- | -------- | ------ |
notbetweenmodelparametersandneuraldynamics(arrow3in
|     |     |     |     |     |     |     | tasks, neural | responses |     | exhibit | remarkably | similar |     | dynamics in |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --------- | --- | ------- | ---------- | ------- | --- | ----------- |
Fig.11)butbetweenmodeldynamicsandneuraldynamicsdirectly
|                  |                  |           |        |                 |       |              | relation | to common | task   | events | and  | behaviors.   | For | example,   |
| ---------------- | ---------------- | --------- | ------ | --------------- | ----- | ------------ | -------- | --------- | ------ | ------ | ---- | ------------ | --- | ---------- |
| (arrow 4         | in Fig. 11). One | example   | is     | a ‘‘two-stage’’ |       | approach, in |          |           |        |        |      |              |     |            |
|                  |                  |           |        |                 |       |              | during a | visual    | search | task   | with | eye movement |     | responses, |
| which behavioral | data             | are first | fitted | and then        | model | dynamics     |          |           |        |        |      |              |     |            |
visualandmotorneuronsdistributedthroughoutparietalcortex
aregeneratedbythebestfittingparameters(Boucheretal.,2007;
Mazurek et al., 2003; Purcell et al., 2010; Purcell, Schall et al., (Thomas & Pare, 2007), prefrontal cortex (Bichot & Schall, 1999;
2012; Ratcliff et al., 2007). The key is that rather than directly Thompson,Hanes,Bichot,&Schall,1996),andsuperiorcolliculus
comparingneuraldynamicstothebest-fittingmodelparameters, (McPeek&Keller,2002;White&Munoz,2011)exhibitresponse
the neural dynamics are compared to the simulated model propertiesconsistentwitharepresentationofperceptualevidence
dynamics. Simulation methods should be chosen to match the ortheaccumulationofevidencetoaresponsethreshold.During
statisticalpoweroftheneuralanalyses;forexample,matchingthe a motion-discrimination task, neurons throughout the same
|     |     |     |     |     |     |     | network | of structures | exhibit |     | dynamics | consistent | with | gradual |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | ------- | --- | -------- | ---------- | ---- | ------- |
numberofsimulatedandobservedtrials,matchingthesimulated
|     |     |     |     |     |     |     | accumulation | of  | sensory | evidence | to  | a threshold | (Ding | & Gold, |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------- | -------- | --- | ----------- | ----- | ------- |
andobservedfiringrates,andsimulatingspikingnoisebasedon
|          |                     |           |     |          |        |         | 2012; Horwitz | &   | Newsome, | 1999; | Kim | & Shadlen, | 1999; | Purcell |
| -------- | ------------------- | --------- | --- | -------- | ------ | ------- | ------------- | --- | -------- | ----- | --- | ---------- | ----- | ------- |
| observed | model trajectories. | Identical |     | measures | should | then be |               |     |          |       |     |            |       |         |
appliedtothemodeldynamicstoallowforadirectcomparisonto & Kiani, 2016; Roitman & Shadlen, 2002). Therefore, while the
observed neural dynamics. This approach avoids many potential strongest conclusions about the decision making process would
pitfalls highlighted by our simulations; for example, incorrect need to include recordings from multiple brain regions, even
assumptions about the direct relationships – or lack thereof – recordingsfromasingleregioncanprovidecriticalinsightsabout
betweenmodelparametersandneuraldynamics.Oneadvantage activitythatistakingplacethroughoutthenetwork.
ofthistwo-stageapproachisthatthemodeldynamicsrepresenta Otherstudieshavesuggestedthatarepresentationofevidence
trueprediction(notafit)totheneuraldata.Ifthesameparameter accumulationmaybemonitorednoninvasivelythroughextracra-
nialelectroencephalography(EEG).Duringcertainperceptualde-
valuesthatmaximizethequalityoffittothebehavioralsoprovide
cisions,extracranialvoltagepotentialscanbeanalyzedtoextract
agoodaprioriaccountoftheunderlyingneuraldynamicswithout
any additional fitting, this provides compelling validation of the anevolvingsignalthatdemonstratesdynamicsconsistentwithev-
modelfitsandguardsagainstoverfitting. idenceaccumulation(Kelly&O’Connell,2013;O’Connell,Dockree,
A complementary approach to link accumulator models to &Kelly,2012).Thelateralizedreadinesspotential,anevent-related
neural data is to replace model mechanisms specified by free potentialassociatedwithmotorpreparation,alsoexhibitsdynam-
parameters with actual neural data. Purcell, Schall et al. (2012) icssimilartoaccumulationtoathreshold(DeJong,Coles,Logan,&
andPurcelletal.(2010)usedtheobservedfiringratesofvisually- Gratton,1990;Gratton,Coles,Sirevaag,Eriksen,&Donchin,1988;
responsive FEF neurons during a visual search task as the input Schurger, Sitt, & Dehaene, 2012). Our conclusions regarding the
toanetworkofmodelaccumulatorsinplaceofthemechanisms relationship between model and neural dynamics should apply

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 169
equallytothesehigher-levelrepresentationsofthedecisionpro- values and applied established neurophysiological measures
cess. The major limitation of larger-scale representations of evi- to model dynamics to test for selective influences of model
denceaccumulationisthattheywillreflectthecombinedactivity parameters on expected neural dynamics. In some cases, model
ofcorticalareasnotdirectlyinvolvedinevidenceaccumulationas parameterscouldbesuccessfullyinferredfrommodeldynamics,
wellasaggregatingoverneuronsrepresentingdistinctchoices.It but in other cases measures of dynamics alone could provide a
isnotclearhowtheseadditionalsignalsmightdistorttheresulting misleading picture about the underlying sources of behavioral
dynamics. variability.Altogether,weargueforamodelingapproachinwhich
bothbehavioralandneuraldataarejointlyconsidered,andmodel
Anessentialgoalforfutureresearchistounderstandhowthe
dynamicsaredirectlycomparedtoneuraldynamics.
representationofevidenceinindividualneuronsscalesuptolarger
populations of neurons within and across multiple areas. Going
Acknowledgments
from individual neurons to larger populations raises important
questionsabouthowevidenceaccumulationiscoordinatedover
This work was supported by grants NIH R01-EY021833 and
neurons and how these different neurons reach a consensus to
the NSF Temporal Dynamics of Learning Center SMA-1041755,
terminate the decision process. Spiking neural network models
andbytheVanderbiltVisionResearchCenter,T32-EYO7135and
havedemonstratedhowlargepoolsofneuronswithrecurrentand P30-EY008126. BP was supported by a postdoctoral fellowship
competitive dynamics could implement the evidence accumula- from the Simons Collaboration on the Global Brain (Award
tion process proposed by these models (Furman & Wang, 2008; number:351208).
Wang, 2002), but the complexity of these models can make it
difficulttocontrastcompetinghypothesesaboutneuronalcoop- References
erationanddecisiontermination.Toaddressthisproblem,Zand-
belt,Purcell,Palmeri,Logan,andSchall(2014)usedsimulationsto Bichot, N. P., & Schall, J. D. (1999). Effects of similarity and history on neural
mechanismsofvisualselection.NatureNeuroscience,2(6),549–554.
explorehowlargeensemblesofredundantaccumulators(multi-
Bogacz,R.,Brown,E.,Moehlis,J.,Holmes,P.,&Cohen,J.D.(2006).Thephysicsof
pleaccumulatorsrepresentingasinglechoice)couldproduceboth optimaldecisionmaking:aformalanalysisofmodelsofperformanceintwo-
RTdistributionsandneuraldynamicsconsistentwithsingle-unit alternativeforced-choicetasks.PsychologicalReview,113(4),700–765.
Bollimunta,A.,&Ditterich,J.(2012).Localcomputationofdecision-relevantnet
recordings.Theyfoundthataverybroadrangeofmechanismsby
sensoryevidenceinparietalcortex.CerebralCortex,22(4),903–917.
whichtheaccumulatorscoordinateandterminatetheirresponse Boucher,L.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2007).Inhibitorycontrol
couldproduceRTsandneuralresponsesconsistentwithobserved inmindandbrain:aninteractiveracemodelofcountermandingsaccades.
PsychologicalReview,114(2),376–397.
behavioral and neural data. On the one hand, these results indi- Brown, S., & Heathcote, A. (2005). A ballistic model of choice response time.
catethatredundantstochasticaccumulatorsrepresentaveryro- PsychologicalReview,112(1),117–128.
bustmethodtoproducerealisticRTdistributionsandsingle-unit Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice
response time: linear ballistic accumulation. Cognitive Psychology, 57(3),
activity.Ontheotherhand,theseresultssuggestthatsingle-unit 153–178.
recordingsandbehavioralonearenotadequatetoidentifythepre- Busemeyer, J. R., & Diederich, A. (2009). Cognitive modeling. Sage Publications,
Incorporated.
cisemechanismsbywhichlargenetworksofaccumulatorscoordi-
Cassey,P.J.,Gaut,G.,Steyvers,M.,&Brown,S.D.(2016).Agenerativejointmodel
nateandterminateuponachoice. forspiketrainsandsaccadesduringperceptualdecision-making.Psychonomic
Distinguishingalternativemechanismsforevidenceaccumula- BulletinandReview.
Churchland,M.M.,Cunningham,J.P.,Kaufman,M.T.,Foster,J.D.,Nuyujukian,P.,
tion in large populations requires simultaneous recordings from Ryu,S.I.,etal.(2012).Neuralpopulationdynamicsduringreaching.Nature,
manyindividualneuronsduringongoingdecisionmaking.Exciting 487(7405),51–56.
Churchland,A.K.,Kiani,R.,Chaudhuri,R.,Wang,X.J.,Pouget,A.,&Shadlen,M.N.
new technological advances in multielectrode arrays have made
(2011).Varianceasasignatureofneuralcomputationsduringdecisionmaking.
itmorecommontorecordsimultaneouslyfromtenstohundreds Neuron,69(4),818–831.
ofneuronsinawakebehavinganimals(Churchland,Yu,Sahani,& Churchland,A.K.,Kiani,R.,&Shadlen,M.N.(2008).Decision-makingwithmultiple
alternatives.NatureNeuroscience,11(6),693–702.
Shenoy,2007).Thusfar,thesetechniqueshaveprimarilybeenap-
Churchland, M. M., Yu, B. M., Sahani, M., & Shenoy, K. V. (2007). Techniques
pliedtothestudyofmotorcontrol(e.g.Churchlandetal.,2012) forextractingsingle-trialactivitypatternsfromlarge-scaleneuralrecordings.
andneuralprosthetics(e.g.Sadtleretal.,2014),butarejuststart- CurrentOpinioninNeurobiology,17(5),609–618.
Cisek,P.(2006).Integratedneuralprocessesfordefiningpotentialactionsand
ingtobeappliedinawakebehavinganimalsduringmorecomplex
deciding between them: a computational model. Journal of Neuroscience,
decision-makingtasks(Cohen&Maunsell,2009;Kianietal.,2014). 26(38),9761–9770.
Inonenotableexample,Kianietal.(2014)trainedalinearclassi- Cohen,M.R.,&Maunsell,J.H.(2009).Attentionimprovesperformanceprimarilyby
fiertodecodeanimal’schoicesfromalargepopulationof∼90neu- reducinginterneuronalcorrelations.NatureNeuroscience,12(12),1594–1600.
Cook,E.P.,&Maunsell,J.H.(2002).Dynamicsofneuronalresponsesinmacaque
ronsrecordedsimultaneouslyfromdorsolateralprefrontalcortex MTandVIPduringmotiondetection.NatureNeuroscience,5(10),985–994.
whileanimalsperformedamotiondirectiondiscriminationtask. DeJong,R.,Coles,M.G.,Logan,G.D.,&Gratton,G.(1990).Insearchofthepointof
noreturn:thecontrolofresponseprocesses.JournalofExperimentalPsychology:
Theaccuracyoftheclassifiergraduallyincreasedthroughoutthe
HumanPerceptionandPerformance,16(1),164.
trialataratethatdependedonthestrengthofsensoryevidence, de Lafuente, V., Jazayeri, M., & Shadlen, M. N. (2015). Representation of
consistentwithapopulation-widerepresentationofevidenceac- accumulating evidence for a decision in two parietal areas. Journal of
Neuroscience,35(10),4306–4318.
cumulation.Thepopulationactivityrevealed‘‘changesofmind’’in DiCarlo,J.J.,&Maunsell,J.H.(2005).Usingneuronallatencytodeterminesensory-
whichtheaccumulatedevidenceflippedfromfavoringonechoice motorprocessingpathwaysinreactiontimetasks.JournalofNeurophysiology,
to another, consistent with predictions of an existing behavioral 93(5),2974–2986.
Ding,L.,&Gold,J.I.(2010).Caudateencodesmultiplecomputationsforperceptual
model(Resulajetal.,2009).Asmorelarge-scaledatasetsbecome decisions.JournalofNeuroscience,30(47),15747–15759.
available,theseapproachescanbefurtherextendedtoconstrain Ding,L.,&Gold,J.I.(2012).Neuralcorrelatesofperceptualdecisionmakingbefore,
potentialsourcesofbehavioralvariabilityatthelevelofindividual during,andafterdecisioncommitmentinmonkeyfrontaleyefield.Cerebral
Cortex,22(5),1052–1067.
trials. Faisal,A.A.,Selen,L.P.,&Wolpert,D.M.(2008).Noiseinthenervoussystem.Nature
ReviewsNeuroscience,9(4),292–303.
Farrell,S.,&Lewandowsky,S.(2010).Computationalmodelingincognition:Principles
4. Conclusions
andpractice.SagePublications,Incorporated.
Fetsch, C. R., Kiani, R., Newsome, W. T., & Shadlen, M. N. (2014). Effects of
Neuraldynamicshaveproventobeapowerfultooltoevaluate corticalmicrostimulationonconfidenceinaperceptualdecision.Neuron,83(4),
797–804.
alternative hypotheses about decision making mechanisms, but Forstmann,B.U.,Dutilh,G.,Brown,S.,Neumann,J.,vonCramon,D.Y.,Ridderinkhof,
theconnectionsbetweenmodelparametersandneuraldynamics K.R.,etal.(2008).Striatumandpre-SMAfacilitatedecision-makingundertime
often go untested. We systematically varied model parameter pressure.ProceedingsoftheNationalAcademyofSciencesoftheUSA,105(45),
17538–17542.

170 B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171
Forstmann,B.U.,Ratcliff,R.,&Wagenmakers,E.J.(2015).Sequentialsampling Monosov,I.E.,&Thompson,K.G.(2009).Frontaleyefieldactivityenhancesobject
modelsincognitiveneuroscience:advantages,applications,andextensions. identificationduringcovertvisualsearch.JournalofNeurophysiology,102(6),
AnnualReviewofPsychology. 3656–3672.
Forstmann,B.U.,Wagenmakers,E.J.,Eichele,T.,Brown,S.,&Serences,J.T.(2011). Moran,R.,Teodorescu,A.R.,&Usher,M.(2015).Postchoiceinformationintegration
Reciprocal relations between cognitive neuroscience and formal cognitive asacausaldeterminantofconfidence:Noveldataandacomputationalaccount.
models:oppositesattract?.TrendsinCognitiveScience,15(6),272–279. CognitivePsychology,78,99–147.
Furman,M.,&Wang,X.J.(2008).Similarityeffectandoptimalcontrolofmultiple- Mulder, M. J., van Maanen, L., & Forstmann, B. U. (2014). Perceptual decision
choicedecisionmaking.Neuron,60(6),1153–1168. neurosciences—amodel-basedreview.Neuroscience,277,872–884.
Gold,J.I.,&Shadlen,M.N.(2000).Representationofaperceptualdecisionin Nawrot,M.P.,Boucsein,C.,RodriguezMolina,V.,Riehle,A.,Aertsen,A.,&Rotter,S.
developingoculomotorcommands.Nature,404(6776),390–394. (2008).Measurementofvariabilitydynamicsincorticalspiketrains.Journalof
Gold,J.I.,&Shadlen,M.N.(2007).Theneuralbasisofdecisionmaking.Annual NeuroscienceMethods,169(2),374–390.
ReviewofNeuroscience,30,535–574. Nosofsky,R.M.,&Palmeri,T.J.(1997).Anexemplar-basedrandomwalkmodelof
Gratton,G.,Coles,M.G.,Sirevaag,E.J.,Eriksen,C.W.,&Donchin,E.(1988).Pre-and speededclassification.PsychologicalReview,104(2),266–300.
poststimulusactivationofresponsechannels:apsychophysiologicalanalysis. O’Connell,R.G.,Dockree,P.M.,&Kelly,S.P.(2012).Asupramodalaccumulation-
JournalofExperimentalPsychology:HumanPerceptionandPerformance,14(3), to-bound signal that determines perceptual decisions in humans. Nature
331–344. Neuroscience,15(12),1729–1735.
Grice,G.R.(1968).Stimulusintensityandresponseevocation.PsychologicalReview, O’Reilly,J.X.,&Mars,R.B.(2011).ComputationalNeuroimaging:LocalisingGreek
75(5),359–373. Letters?CommentonForstmann‘‘etal.TrendsinCognitiveSciences,15(10).
Grice,G.R.(1972).Applicationofavariablecriterionmodeltoauditoryreaction
Osman,A.,Lou,L.,Muller-Gethmann,H.,Rinkenauer,G.,Mattes,S.,&Ulrich,R.
timeasafunctionofthetypeofcatchtrial.PerceptionandPsychophysics,12(1),
(2000).Mechanismsofspeed-accuracytradeoff:evidencefromcovertmotor
103–107.
processes.BiologicalPsychology,51(2–3),173–199.
Hanes,D.P.,&Schall,J.D.(1996).Neuralcontrolofvoluntarymovementinitiation.
Palmer,J.,Huk,A.C.,&Shadlen,M.N.(2005).Theeffectofstimulusstrengthonthe
Science,274(5286),427–430.
speedandaccuracyofaperceptualdecision.JournalofVision,5(5),376–404.
Hanks,T.D.,Ditterich,J.,&Shadlen,M.N.(2006).Microstimulationofmacaque
area LIP affects decision-making in a motion discrimination task. Nature Palmeri,T.J.(1997).Exemplarsimilarityandthedevelopmentofautomaticity.
Neuroscience,9(5),682–689. Journal of Experimental Psychology. Learning, Memory, and Cognition, 23(2),
Hanks,T.,Kiani,R.,&Shadlen,M.N.(2014).Aneuralmechanismofspeed-accuracy 324–354.
Palmeri,T.J.,Schall,J.D.,&Logan,G.D.(2014).Oxfordhandbookofcomputationaland
tradeoffinmacaqueareaLIP.Elife,3.
mathematicalpsychology,Neurocognitivemodelingofperceptualdecisionmaking.
Hawkins,G.E.,Forstmann,B.U.,Wagenmakers,E.J.,Ratcliff,R.,&Brown,S.D.
Petrov,A.A.,VanHorn,N.M.,&Ratcliff,R.(2011).Dissociableperceptual-learning
(2015).Revisitingtheevidenceforcollapsingboundariesandurgencysignals
mechanismsrevealedbydiffusion-modelanalysis.PsychonomicBulletinand
inperceptualdecision-making.JournalofNeuroscience,35(6),2476–2484.
Review,18(3),490–497.
Heekeren, H. R., Marrett, S., Bandettini, P. A., & Ungerleider, L. G. (2004). A
generalmechanismforperceptualdecision-makinginthehumanbrain.Nature, Pleskac,T.J.,&Busemeyer,J.R.(2010).Two-stagedynamicsignaldetection:a
431(7010),859–862. theoryofchoice,decisiontime,andconfidence.PsychologicalReview,117(3),
Heitz,R.P.,&Schall,J.D.(2012).Neuralmechanismsofspeed-accuracytradeoff. 864–901.
Pouget, P., Logan, G. D., Palmeri, T. J., Boucher, L., Pare, M., & Schall, J. D.
Neuron,76(3),616–628.
(2011). Neural basis of adaptive response time adjustment during saccade
Horwitz,G.D.,&Newsome,W.T.(1999).Separatesignalsfortargetselection
countermanding.JournalofNeuroscience,31(35),12604–12612.
and movement specification in the superior colliculus. Science, 284(5417),
1158–1161. Purcell,B.A.,Heitz,R.P.,Cohen,J.Y.,&Schall,J.D.(2012).Responsevariability
Jones,M.,&Dzhafarov,E.N.(2014).Unfalsifiabilityandmutualtranslatabilityof of frontal eye field neurons modulates with sensory input and saccade
majormodelingschemesforchoicereactiontime.PsychologicalReview,121(1), preparationbutnotvisualsearchsalience.JournalofNeurophysiology,108(10),
1–32. 2737–2750.
Kelly,S.P.,&O’Connell,R.G.(2013).Internalandexternalinfluencesontherate Purcell,B.A.,Heitz,R.P.,Cohen,J.Y.,Schall,J.D.,Logan,G.D.,&Palmeri,T.J.(2010).
ofsensoryevidenceaccumulationinthehumanbrain.JournalofNeuroscience, Neurallyconstrainedmodelingofperceptualdecisionmaking.Psychological
33(50),19434–19441. Review,117(4),1113–1143.
Kiani,R.,Cueva,C.J.,Reppas,J.B.,&Newsome,W.T.(2014).Dynamicsofneural Purcell,B.A.,&Kiani,R.(2016).Neuralmechanismsofpost-erroradjustmentsof
populationresponsesinprefrontalcortexindicatechangesofmindonsingle decisionpolicyinparietalcortex.Neuron,89(3),658–671.
trials.CurrentBiology,24(13),1542–1547. Purcell,B.A.,Schall,J.D.,Logan,G.D.,&Palmeri,T.J.(2012).Fromsalienceto
Kim,J.N.,&Shadlen,M.N.(1999).Neuralcorrelatesofadecisioninthedorsolateral saccades:multiple-alternativegatedstochasticaccumulatormodelofvisual
prefrontalcortexofthemacaque.NatureNeuroscience,2(2),176–185. search.JournalofNeuroscience,32(10),3433–3446.
Laming,D.(1968).Informationtheoryofchoice-reactiontimes.NewYork:Academic Ratcliff,R.(1978).Atheoryofmemoryretrieval.PsychologicalReview,85(2),59.
Press. Ratcliff, R. (2013). Parameter variability and distributional assumptions in the
Link,S.W.(1992).Thewavetheoryofdifferenceandsimilarity.Hillsdale:Lawrence diffusionmodel.PsychologicalReview,120(1),281–292.
ErlbaumAssociates.
Ratcliff,R.,Cherian,A.,&Segraves,M.(2003).Acomparisonofmacaquebehavior
Link,S.,&Heath,R.(1975).Asequentialtheoryofpsychologicaldiscrimination.
andsuperiorcolliculusneuronalactivitytopredictionsfrommodelsoftwo-
Psychometrika,40(1),77–105.
choicedecisions.JournalofNeurophysiology,90(3),1392–1407.
Lo,C.C.,&Wang,X.J.(2006).Cortico-basalgangliacircuitmechanismforadecision
Ratcliff, R., Hasegawa, Y. T., Hasegawa, R. P., Smith, P. L., & Segraves, M. A.
thresholdinreactiontimetasks.NatureNeuroscience,9(7),956–963.
(2007).Dualdiffusionmodelforsingle-cellrecordingdatafromthesuperior
Lo,C.C.,Wang,C.T.,&Wang,X.J.(2015).Speed-accuracytradeoffbyacontrolsignal
colliculusinabrightness-discriminationtask.JournalofNeurophysiology,97(2),
with balanced excitation and inhibition. Journal of Neurophysiology, 114(1),
1756–1774.
650–661.
Logan,G.D.,&Cowan,W.B.(1984).Ontheabilitytoinhibitthoughtandaction:A Ratcliff,R.,&McKoon,G.(2008).Thediffusiondecisionmodel:theoryanddatafor
theoryofanactofcontrol.PsychologicalReview,91(3),295. two-choicedecisiontasks.NeuralComputation,20(4),873–922.
Logan,G.D.,Yamaguchi,M.,Schall,J.D.,&Palmeri,T.J.(2015).Inhibitorycontrol Ratcliff,R.,&Rouder,J.N.(1998).Modelingresponsetimesfortwo-choicedecisions.
in mind and brain 2.0: blocked-input models of saccadic countermanding. PsychologicalScience,9(5),347–356.
PsychologicalReview,122(2),115–147. Ratcliff,R.,&Smith,P.L.(2004).Acomparisonofsequentialsamplingmodelsfor
Mante,V.,Sussillo,D.,Shenoy,K.V.,&Newsome,W.T.(2013).Context-dependent two-choicereactiontime.PsychologicalReview,111(2),333–367.
computationbyrecurrentdynamicsinprefrontalcortex.Nature,503(7474), Ratcliff,R.,&Smith,P.L.(2010).Perceptualdiscriminationinstaticanddynamic
78–84. noise: the temporal relation between perceptual encoding and decision
Mazurek,M.E.,Roitman,J.D.,Ditterich,J.,&Shadlen,M.N.(2003).Aroleforneural making.JournalofExperimentalPsychology:General,139(1),70–94.
integratorsinperceptualdecisionmaking.CerebralCortex,13(11),1257–1269. Ratcliff,R.,Thapar,A.,&McKoon,G.(2006).Aging,practice,andperceptualtasks:
McPeek, R. M., & Keller, E. L. (2002). Saccade target selection in the superior adiffusionmodelanalysis.PsychologyandAging,21(2),353–371.
colliculus during a visual search task. Journal of Neurophysiology, 88(4), Reddi,B.A.,&Carpenter,R.H.(2000).Theinfluenceofurgencyondecisiontime.
2019–2034. NatureNeuroscience,3(8),827–830.
Meyer, D. E., Osman, A. M., Irwin, D. E., & Yantis, S. (1988). Modern mental Resulaj,A.,Kiani,R.,Wolpert,D.M.,&Shadlen,M.N.(2009).Changesofmindin
chronometry.BiologicalPsychology,26(1–3),3–67. decision-making.Nature,461(7261),263–266.
Miller,J.(1982).Discreteversuscontinuousstagemodelsofhumaninformation
Roitman, J. D., & Shadlen, M. N. (2002). Response of neurons in the lateral
processing: in search of partial output. Journal of Experimental Psychology:
intraparietalareaduringacombinedvisualdiscriminationreactiontimetask.
HumanPerceptionandPerformance,8(2),273–296.
JournalofNeuroscience,22(21),9475–9489.
Miller,J.(1998).Effectsofstimulus–responseprobabilityonchoicereactiontime:
Sadtler,P.T.,Quick,K.M.,Golub,M.D.,Chase,S.M.,Ryu,S.I.,Tyler-Kabara,E.C.,
Evidence from the lateralized readiness potential. Journal of Experimental
etal.(2014).Neuralconstraintsonlearning.Nature,512(7515),423–426.
Psychology:HumanPerceptionandPerformance,24(5),1521.
Schall,J.D.(2004).Onbuildingabridgebetweenbrainandbehavior.AnnualReview
Miller, J., Ulrich, R., & Rinkenauer, G. (1999). Effects of stimulus intensity on
ofPsychology,55,23–50.
thelateralizedreadinesspotential.JournalofExperimentalPsychology:Human
Schall,J.D.(2013).Macrocircuits:decisionnetworks.CurrentOpinioninNeurobiol-
PerceptionandPerformance,25(5),1454.
ogy,23(2),269–274.

B.A.Purcell,T.J.Palmeri/JournalofMathematicalPsychology76(2017)156–171 171
Schurger,A.,Sitt,J.D.,&Dehaene,S.(2012).Anaccumulatormodelforspontaneous Thura,D.,Cos,I.,Trung,J.,&Cisek,P.(2014).Context-dependenturgencyinfluences
neuralactivitypriortoself-initiatedmovement.ProceedingsoftheNational speed-accuracy trade-offs in decision-making and movement execution.
AcademyofSciencesoftheUSA,109(42),E2904–2913. JournalofNeuroscience,34(49),16442–16454.
Scudder,C.A.,Kaneko,C.S.,&Fuchs,A.F.(2002).Thebrainstemburstgenerator Townsend,J.T.,&Ashby,F.G.(1983).Stochasticmodelingofelementarypsychological
forsaccadiceyemovements:amodernsynthesis.ExperimentalBrainResearch, processes.CambridgeUniversityPress.
142(4),439–462. Tuerlinckx, F., Maris, E., Ratcliff, R., & De Boeck, P. (2001). A comparison of
Shadlen,M.N.,Hanks,T.D.,Churchland,A.K.,Kiani,R.,&Yang,T.(2006).The fourmethodsforsimulatingthediffusionprocess.BehaviorResearchMethods,
speedandaccuracyofasimpleperceptualdecision:amathematicalprimer. Instruments,&Computers,33(4),443–456.
InBayesianrain:Probabilisticapproachestoneuralcoding(pp.209–237). Turner,B.M.,Forstmann,B.U.,Love,B.,Palmeri,T.J.,&VanMaanen,L.(2017).
Shadlen,M.N.,&Kiani,R.(2013).Decisionmakingasawindowoncognition. Approaches to analysis in model-based cognitive neuroscience. Journal of
Neuron,80(3),791–806. MathematicalPsychology,76,65–79.
Smith,P.L.(1995).Psychophysicallyprincipledmodelsofvisualsimplereaction Turner,B.M.,vanMaanen,L.,&Forstmann,B.U.(2015).Informingcognitiveab-
time.PsychologicalReview,102(3),567. stractionsthroughneuroimaging:theneuraldriftdiffusionmodel.Psychologi-
Smith,P.L.(2010).FromPoissonshotnoisetotheintegratedOrnstein–Uhlenbeck calReview,122(2),312–336.
process:Neurallyprincipledmodelsofinformationaccumulationindecision- Usher,M.,&McClelland,J.L.(2001).Thetimecourseofperceptualchoice:theleaky,
makingandresponsetime.JournalofMathematicalPsychology,54(2),266–283. competingaccumulatormodel.PsychologicalReview,108(3),550–592.
Smith,P.L.(2015).ThePoissonshotnoisemodelofvisualshort-termmemoryand Vandekerckhove,J.,&Tuerlinckx,F.(2007).FittingtheRatcliffdiffusionmodelto
choiceresponsetime:Normalizedcodingbyneuralpopulationsize.Journalof experimentaldata.PsychonomicBulletinandReview,14(6),1011–1026.
MathematicalPsychology,66,41–52. vanMaanen,L.,Brown,S.D.,Eichele,T.,Wagenmakers,E.J.,Ho,T.,Serences,J.,
Smith,P.L.,&Ratcliff,R.(2004).Psychologyandneurobiologyofsimpledecisions. etal.(2011).Neuralcorrelatesoftrial-to-trialfluctuationsinresponsecaution.
TrendsinNeuroscience,27(3),161–168. JournalofNeuroscience,31(48),17488–17495.
Smith, P. L., & Vickers, D. (1988). The accumulator model of two-choice Vickers,D.(1979).Decisionprocessesinvisualperception.NewYork:AcademicPress.
discrimination.JournalofMathematicalPsychology,32(2),135–168. Wagenmakers,E.J.,Ratcliff,R.,Gomez,P.,&McKoon,G.(2008).Adiffusionmodel
Smulders,F.T.,Kok,A.,Kenemans,J.L.,&Bashore,T.R.(1995).Thetemporal accountofcriterionshiftsinthelexicaldecisiontask.JournalofMemoryand
selectivityofadditivefactoreffectsonthereactionprocessrevealedinERP Language,58(1),140–159.
componentlatencies.ACTAPsychologica(Amsterdam),90(1–3),97–109. Wang,X.J.(2002).Probabilisticdecisionmakingbyslowreverberationincortical
Stuphorn,V.,&Schall,J.D.(2006).Executivecontrolofcountermandingsaccades circuits.Neuron,36(5),955–968.
bythesupplementaryeyefield.NatureNeuroscience,9(7),925–931. White,C.N.,Mumford,J.A.,&Poldrack,R.A.(2012).Perceptualcriteriainthe
Teller,D.Y.(1984).Linkingpropositions.VisionResearch,24(10),1233–1246. humanbrain.JournalofNeuroscience,32(47),16716–16724.
Thomas, N. W., & Pare, M. (2007). Temporal processing of saccade targets in White, B. J., & Munoz, D. P. (2011). Separate visual signals for saccade
parietalcortexareaLIPduringvisualsearch.JournalofNeurophysiology,97(1), initiationduringtargetselectionintheprimatesuperiorcolliculus.Journalof
942–947. Neuroscience,31(5),1570–1578.
Thompson,K.G.,Hanes,D.P.,Bichot,N.P.,&Schall,J.D.(1996).Perceptualand Woodman,G.F.,Kang,M.S.,Thompson,K.,&Schall,J.D.(2008).Theeffectof
motorprocessingstagesidentifiedintheactivityofmacaquefrontaleyefield visualsearchefficiencyonresponsepreparation:neurophysiologicalevidence
neuronsduringvisualsearch.JournalofNeurophysiology,76(6),4040–4055. fordiscreteflow.PsychologicalScience,19(2),128–136.
Thura, D., & Cisek, P. (2014). Deliberation and commitment in the premotor Zandbelt,B.,Purcell,B.A.,Palmeri,T.J.,Logan,G.D.,&Schall,J.D.(2014).Response
andprimarymotorcortexduringdynamicdecisionmaking.Neuron,81(6), timesfromensemblesofaccumulators.ProceedingsoftheNationalAcademyof
1401–1416. SciencesoftheUnitedStatesofAmerica,111(7),2848–2853.