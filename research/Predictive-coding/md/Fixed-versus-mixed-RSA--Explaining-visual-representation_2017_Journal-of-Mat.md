JournalofMathematicalPsychology76(2017)184–197
ContentslistsavailableatScienceDirect
Journal of Mathematical Psychology
journalhomepage:www.elsevier.com/locate/jmp
Fixed versus mixed RSA: Explaining visual representations by fixed
and mixed feature sets from shallow and deep computational models
Seyed-MahdiKhaligh-Razavia,b,∗ ,LindaHenrikssona,c,KendrickKayd,
NikolausKriegeskortea
aMRCCognitionandBrainSciencesUnit,Cambridge,UK
bComputerScience&ArtificialintelligenceLaboratory,MassachusettsInstituteofTechnology,Cambridge,MA,USA
cDepartmentofNeuroscienceandBiomedicalEngineering,AaltoUniversity,Espoo,Finland
dDepartmentofPsychology,WashingtonUniversityinSt.Louis,St.Louis,MO,USA
h i g h l i g h t s
• Wetestedcomputationalmodelsofrepresentationsinventral-streamvisualareas.
• Wecomparedrepresentationaldissimilaritieswith/withoutlinearremixingofmodelfeatures.
• Earlyvisualareaswerebestexplainedbyshallow–andhigherbydeep–models.
• Unsupervisedshallowmodelsperformedbetterwithoutlinearremixingoftheirfeatures.
• Asuperviseddeepconvolutionalnetperformedbestwithlinearfeatureremixing.
a r t i c l e i n f o a b s t r a c t
Articlehistory: Studies of the primate visual system have begun to test a wide range of complex computational
Availableonline19November2016 object-visionmodels.Realisticmodelshavemanyparameters,whichinpracticecannotbefittedusing
the limited amounts of brain-activity data typically available. Task performance optimization (e.g.
Keywords:
usingbackpropagationtotrainneuralnetworks)providesmajorconstraintsforfittingparametersand
Representationalsimilarityanalysis
discoveringnonlinearrepresentationalfeaturesappropriateforthetask(e.g.objectclassification).Model
MixedRSA
representationscanbecomparedtobrainrepresentationsintermsoftherepresentationaldissimilarities
Voxel-receptive-fieldmodelling
Object-visionmodels theypredictforanimageset.Thismethod,calledrepresentationalsimilarityanalysis(RSA),enablesus
Deepconvolutionalnetworks totesttherepresentationalfeaturespaceasis(fixedRSA)ortofitalineartransformationthatmixesthe
nonlinearmodelfeaturessoastobestexplainacorticalarea’srepresentationalspace(mixedRSA).Like
voxel/population-receptive-fieldmodelling,mixedRSAusesatrainingset(differentstimuli)tofitone
weightpermodelfeatureandresponsechannel(voxelshere),soastobestpredicttheresponseprofile
acrossimagesforeachresponsechannel.Weanalysedresponsepatternselicitedbynaturalimages,which
weremeasuredwithfunctionalmagneticresonanceimaging(fMRI).Wefoundthatearlyvisualareas
werebestaccountedforbyshallowmodels,suchasaGaborwaveletpyramid(GWP).TheGWPmodel
performedsimilarlywithandwithoutmixing,suggestingthattheoriginalfeaturesalreadyapproximated
the representational space, obviating the need for mixing. However, a higher ventral-stream visual
representation(lateraloccipitalregion)wasbestexplainedbythehigherlayersofadeepconvolutional
networkandmixingofitsfeaturesetwasessentialforthismodeltoexplaintherepresentation.We
suspectthatmixingwasessentialbecausetheconvolutionalnetworkhadbeentrainedtodiscriminatea
setof1000categories,whosefrequenciesinthetrainingsetdidnotmatchtheirfrequenciesinnatural
experience or their behavioural importance. The latter factors might determine the representational
prominenceofsemanticdimensionsinhigher-levelventral-streamareas.Ourresultsdemonstratethe
∗
Correspondingauthorat:ComputerScience&ArtificialintelligenceLaboratory,MassachusettsInstituteofTechnology,Cambridge,MA,USA.
E-mailaddress:skhaligh@mit.edu(S.-M.Khaligh-Razavi).
http://dx.doi.org/10.1016/j.jmp.2016.10.007
0022-2496/©2016TheAuthor(s).PublishedbyElsevierInc.ThisisanopenaccessarticleundertheCCBY-NC-NDlicense(http://creativecommons.org/licenses/by-nc-nd/
4.0/).

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 185
benefitsoftestingboththespecificrepresentationalhypothesisexpressedbyamodel’soriginalfeature
spaceandthehypothesisspacegeneratedbylineartransformationsofthatfeaturespace.
©2016TheAuthor(s).PublishedbyElsevierInc.
ThisisanopenaccessarticleundertheCCBY-NC-NDlicense
(http://creativecommons.org/licenses/by-nc-nd/4.0/).
1. Introduction RSAmakesiteasytotest‘‘fixed’’models,thatis,modelsthat
havenofreeparameterstobefitted.Fixedmodelscanbeobtained
by optimizing parameters for task performance (Krizhevsky,
Sensoryprocessingisthoughttorelyonasequenceoftrans-
Sutskever,&Hinton,2012;LeCun,Bengio,&Hinton,2015;Yamins
formationsoftheinput.Ateachstage,aneuronalpopulation-code
et al., 2014). This approach is essential, because realistic models
re-representstherelevantinformationinaformatmoresuitable
ofbraininformationprocessinghavelargenumbersofparameters
for subsequent braincomputations that ultimately contribute to
(reflectingthesubstantialdomainknowledgerequiredforfeatsof
adaptivebehaviour.Thechallengeforcomputationalneuroscience
intelligence),andbraindataarecostlyandlimited.However,we
istobuildmodelsthatperformsuchtransformationsoftheinput
may still want to adjust our models on the basis of brain data.
andtotestthesemodelswithbrain-activitydata.
Forexample,itmightbethatourmodelcontainsallthenonlinear
Herewetestawiderangeofcandidatecomputationalmodels
features needed to perfectly explain a brain representation, but
of the representations along the ventral visual stream, which
inthewrongproportions:withthebraindevotingmoreneurons
is thought to enable object recognition. The ventral stream
to some representational features than to others. Alternatively,
culminates in the inferior temporal cortex, which has been
some features might have greater gain than others in the brain
intensively studied in primates (Bell, Hadj-Bouziane, Frihauf,
representation.Bothoftheseeffectscanbemodelledbyassigning
Tootell, & Ungerleider, 2009; Hung, Kreiman, Poggio, & DiCarlo,
aweighttoeachfeature(Fig.1,upperright;Jozwik,Kriegeskorte,
2005; Kriegeskorte et al., 2008a) and humans (e.g. Haxby
& Mur, 2015; Khaligh-Razavi & Kriegeskorte, 2014). If a fixed
et al., 2001; Huth, Nishimoto, Vu, & Gallant, 2012; Kanwisher,
model’s RDM does not match the brain RDM, it is important to
McDermott,&Chun,1997).Therepresentationinthishighervisual
findoutwhetheritisjustthefeatureweightingthatiscausingthe
areaistheresultofcomputationsperformedinstagesacrossthe
mismatch.
hierarchy of the visual system. There has been good progress in
Herewetakeastepbeyondrepresentationalweightingandex-
understanding and modelling early visual areas (e.g. Eichhorn,
ploreahigher-parametricwaytofittherepresentationalspaceofa
Sinz, & Bethge, 2009; Güçlü & van Gerven, 2014; Hegdé & Van
computationalmodeltobraindata.Ourtechniquetakesadvantage
Essen, 2000; Kay, Winawer, Rokem, Mezer, & Wandell, 2013),
ofvoxel/cell-populationreceptive-field(RF)modelling(Dumoulin
and increasingly also intermediate (e.g. V4) and higher ventral-
andWandell,2008;Huthetal.,2012;Kayetal.,2008)tolinearly
stream areas (e.g. Cadieu et al., 2014; Grill-Spector & Weiner,
mixmodelfeaturesandmapthemtovoxelresponses.Representa-
2014;Güçlü&Gerven,2015;Khaligh-Razavi&Kriegeskorte,2014;
tionalmixingallowsarbitrarylineartransformations(Fig.1,lower
Kriegeskorte,2015;Pasupathy&Connor,2002;Yaminsetal.,2014;
right). Whereas representational weighting involves fitting just
Ziemba&Freeman,2015).HereweusedatafromKay,Naselaris,
oneweightforeachunit,representationalmixinginvolvesfitting
Prenger,andGallant(2008)totestthewiderangeofcomputational
oneweightforeachunitforeachresponsechannel,whereweight-
modelsfromKhaligh-RazaviandKriegeskorte(2014)onmultiple
ingcanstretchorsqueezethespacealongitsoriginalaxes,mixing
visualareasalongtheventralstream.Inaddition,wecombinethe
canstretchandsqueezealsoinobliquedirections,androtateand
fitting of linear models used in voxel-receptive-field modelling
shearthespaceaswell.Inparticular,itcancomputedifferencesbe-
Kay et al. (2008) with tests of model performance at the level
tweentheoriginalfeatures.HereweintroducemixedRSA,inwhich
of representational dissimilarities (Kriegeskorte & Kievit, 2013;
alinearremixingofthemodelfeaturesisfirstlearntusingatrain-
Kriegeskorte,Mur,&Bandettini,2008b;Nilietal.,2014).
ingdataset,soastobestexplainthebrainresponsepatterns.
The geometry of a representation can be usefully character-
Voxel-RFmodellingfitsalineartransformationofthefeatures
izedbyarepresentationaldissimilaritymatrix(RDM)computedby
of a computational model to predict a given voxel’s response.
comparingthepatternsofbrainactivityelicitedbyasetofvisual
We bring RSA and voxel-RF modelling together by constructing
stimuli.Tomotivatethischaracterization,considerthecasewhere
RDMs based on voxel response patterns predicted by voxel-RF
dissimilaritiesaremeasuredasEuclideandistances.TheRDMthen
models.Modelfeaturesarefirstmappedtothebrainspace(asin
completelydefinestherepresentationalgeometry.Tworepresen-
voxel-RF modelling) and the predicted and measured RDMs are
tationsthathavethesameRDMmightdifferinthewaytheunits
thenstatisticallycompared(asinRSA).Weusethelinearmodel
sharethejobofrepresentingthestimulusspace.However,thetwo
to predict measured response patterns for a test set of stimuli
representations would contain the same information and, down
thathavenotbeenusedinlearningthelinearremixing.Wethen
toanorthogonallineartransform,inthesameformat.Assuming
compare the RDMs for the actual measured response patterns
thatthenoiseisisotropic,alinearorradial-basisfunctionreadout
to RDMs for the response patterns predicted with and without
mechanismcouldaccessallthesamefeaturesineachofthetwo
linearremixing.Thisapproachenablesustotest(a)theparticular
representations,andatthesamesignal-to-noiseratio.
representational hypothesis of each computational model and
Intheframeworkofrepresentationalsimilarityanalysis(RSA),
(b)thehypothesisspacegeneratedbylineartransformationsofthe
representations can be compared between model layers and
model’scomputationalfeatures.
brain areas by computing the correlation between their RDMs
(Kriegeskorte, 2009; Nili et al., 2014). Each RDM contains a
representational dissimilarity for each pair of stimulus-related 2. Methods
responsepatterns(Kriegeskorte&Kievit,2013;Kriegeskorteetal.,
2008b). We use the RSA framework here to compare processing Invoxelreceptive-fieldmodelling,alinearcombinationofthe
stagesincomputationalmodelswiththestagesofprocessingin modelfeaturesisfittedusingatrainingsetandresponse-pattern
thehierarchyofventralvisualpathway. prediction performance is assessed on a separate test set with

186 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
(2008) (Fig. 2(A)). We then predict the response patterns for a
set of test stimuli (120 images). Finally, we use RSA to compare
pattern-dissimilaritiesbetweenthepredictedandmeasuredvoxel
responsesforthe120testimages(Fig.2(B)).Thevoxel-RFfitting
is a way of mixing the model features so as to better predict
brain responses. By mixing model features we can investigate
the possibility that all essential nonlinearities are present in a
model,andtheyjustneedtobeappropriatelylinearlycombined
toapproximatetherepresentationalgeometryofagivencortical
area. By linear mixing of features (affine transformation of the
model features), we go beyond stretching and squeezing the
representational space along its original axes (Khaligh-Razavi &
Fig.1. Thetransformationoftherepresentationalspaceresultingfromweighting
andmixingofmodelfeatures.Thefeaturesofamodelspanarepresentationalspace Kriegeskorte,2014)andattempttocreatenewfeaturesaslinear
(left).Thefigureshowsacartoon2-dimensionalmodel-featurespace.Weightingof combinations of the original features. This affine linear recoding
thefeatures(right,top)amountstostretchingandsqueezingoftherepresentational provides a more general transformation, which includes feature
spacealongitsoriginalfeaturedimensions.Mixing(right,bottom)constitutesa
weightingasaspecialcase.
moregeneralclassoftransformations.Weusethetermmixingtodenoteanylinear
transformation,includingrotation,stretchingandsqueezingalongarbitraryaxes, Training: During the training phase (Fig. 2(A)), for each of
andshearing. the brain voxels we learn a weight vector and an offset value
that maps the internal representation of an object-vision model
responsestodifferentimages(Cowen,Chun,&Kuhl,2014;Ester, to the responses of brain voxels. The offset is a constant value
Sprague,&Serences,2015;Kayetal.,2008;Mitchelletal.,2008; thatislearntinthetrainingphaseandisthenaddedtothesum
Naselaris, Kay, Nishimoto, & Gallant, 2011; Sprague & Serences, of the weighted voxel responses. One offset is learnt per voxel.
2013). This method typically requires a large training data set Weonlyusethe1750trainingimagesandthevoxelresponsesto
andalsopriorassumptionsontheweightstopreventoverfitting, these stimuli. The weights, and the offset value are determined
especiallywhenmodelshavemanyrepresentationalfeatures. bygradientdescentwithearlystopping.Earlystoppingisaform
An alternative method is representational similarity analysis of regularization (Skouras, Goutis, & Bramson, 1994), where the
(RSA)(Kriegeskorte,2009;Kriegeskorte&Kievit,2013;Kriegesko- magnitude of model parameter estimates is shrunk in order to
rteetal.,2008b;Nilietal.,2014).RSAcanrelaterepresentations preventoverfitting.Anewmappingfrommodelfeaturestobrain
fromdifferentsources(e.g.computationalmodelsandfMRIpat- voxelsislearntforeachoftheobject-visionmodels.
terns)bycomparingtheirrepresentationaldissimilarities.Therep- Regularizationdetails:Weusedtheregularizationsuggested
resentational dissimilarity matrix (RDM) is a square symmetric by Skouras et al. (1994), where the shrinkage estimator of the
matrix,inwhichthediagonalentriesreflectcomparisonsbetween parametersismotivatedbythegradient-descentalgorithmused
identicalstimuliandare0,bydefinition.Eachoff-diagonalvalue tominimizethesumofsquarederrors(thereforeanL2penalty).
indicatesthedissimilaritybetweentheactivitypatternsassociated Theregularizationresultsfromearlystoppingofthealgorithm.The
withtwodifferentstimuli.Intuitively,anRDMencapsulateswhat algorithmstopswhenitencountersaseriesofiterationsthatdo
distinctionsbetweenstimuliareemphasizedandwhatdistinctions notimproveperformanceontheestimationset.Stoppingtimeis
arede-emphasizedintherepresentation.Inthisstudy,thefMRIre- afreeparameterthatissetusingcross-validation.Anearlierstop
sponsepatternsevokedbythedifferentnaturalimagesformedthe meansgreaterregularization.Theregularizationinducedbyearly
basisofrepresentationaldissimilaritymatrices(RDMs).Themea- stoppinginthecontextofgradientdescenttendstokeepthesizes
surefordissimilaritywascorrelationdistance(1—Pearsonlinear of weights small (and tends to not break correlations between
correlation)betweentheresponsepatterns.WeusedtheRSATool- parameters).Skourasetal.(1994)showthatearlystoppingwith
box(Nilietal.,2014). gradientdescentisverysimilartotheregularizationgivenbyridge
The advantage of RSA is that the model representations can regression,whichisaL2penalty.
readilybecomparedwiththebraindata,withouthavingtofita Testing: In the testing phase (Fig. 2(B)), we use the learned
linearmappingfromthecomputationalfeaturestothemeasured mappingtopredictvoxelresponsestothe120teststimuli.Fora
responses.Assumingthemodelhasnofreeparameterstobeset givenmodelandapresentedimage,weusetheextractedmodel
usingthebrain-activitydata,notrainingsetofbrain-activitydatais featuresandcalculatetheinnerproductofthefeaturevectorwith
neededandwedonotneedtoworryaboutoverfittingtothebrain- eachoftheweightvectorsthatwerelearntinthetrainingphase
activitydata.However,ifthesetofnonlinearfeaturescomputed foreachvoxel.Wethenaddthelearntoffsetvaluetotheresults
by the model is correct, but their relative prominence or linear of the inner product for each voxel. This gives us the predicted
combinationisincorrectforexplainingthebrainrepresentation, voxel responses to the presented image. The same procedure is
classicRSA(i.e.fixedRSA)maygivenoindicationthatthemodel’s repeatedforalltheteststimuli.ThenanRDMisconstructedusing
featurescanbelinearlyrecombinedtoexplaintherepresentation. thepairwisedissimilaritiesbetweenpredictedvoxelresponsesto
Receptive-fieldmodellingmustfitmanyparametersinorderto theteststimuli.
comparerepresentationsbetweenbrainsandmodels.ClassicRSA Advantages over fixed RSA and voxel receptive-field mod-
fits no parameters, testing fixed models without using the data elling: Considering the predictive performance of either (a) the
tofitanyaspectoftherepresentationalspace.Herewecombine particularsetoffeaturesofamodel(fixedRSA)or(b)lineartrans-
elementsofthetwomethods:Wefitlinearpredictionmodelsand formations of the model (voxel-RF modelling) provides ambigu-
thenstatisticallycomparepredictedrepresentationaldissimilarity ous results. In fixed RSA (Fig. 3(A)), it remains unclear to what
matrices. extentfittingalineartransformationmightimproveperformance.
Invoxel-RFmodelling,itremainsunclearwhetherthesetoffea-
2.1. MixedRSA:combiningvoxel-receptive-fieldmodellingwithRSA turesofthemodel,asis,alreadyspansthecorrectrepresentational
space.MixedRSA(Fig.3(B))enablesustocomparefittedandun-
Usingvoxel-RFmodelling,wefirstfitalinearmappingbetween fittedvariantsofeachmodel.
model representations and each of the brain voxels based on a Fitted linear feature combinations may not explain the brain
trainingdataset(voxelresponsesfor1750images)fromKayetal. data in voxel-RF modelling for a combination of three reasons:

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 187
Fig.2. Fittingalinearmodeltomixrepresentationalmodelfeatures.(A)Training:learningreceptivefieldmodelsthatmapmodelfeaturestobrainvoxelresponses.There
isonereceptivefieldmodelforeachvoxel.Ineachreceptivefieldmodel,theweightvectorandtheoffsetarelearntinatrainingphaseusing1750trainingimages,forwhich
wehadmodelfeaturesandvoxelresponses.Theweightsaredeterminedbygradientdescentwithearlystopping.Thefigureshowstheprocessforasamplemodel(e.g.gist
features);thesametraining/testingprocesswasdoneforeachoftheobject-visionmodels.Theoffsetisaconstantvaluethatislearnedinthetrainingphaseandisthen
addedtothesumoftheweightedvoxelresponses.Oneoffsetislearntpervoxel.(B)Testing:predictingvoxelresponsesusingmodelfeaturesextractedfromanimage.In
thetestingphase,weused120testimages(notincludedinthetrainingimages).Foreachimage,modelfeatureswereextractedandresponsesforeachvoxelwerepredicted
usingthereceptivefieldmodelslearnedinthetrainingphase.Thenarepresentationaldissimilaritymatrix(RDM)isconstructedusingthepairwisedissimilaritiesbetween
predictedvoxelresponsestotheteststimuli.
(1) the features do not provide a sufficient basis, (2) the linear this suggests that the original feature space provides a better
model suffers from overfitting, (3) the prior implicit to the estimateofrelativeprominenceandlinearmixingofthefeatures
regularizationprocedurepreventsfindingpredictiveparameters. thanthefittingprocedurecanprovide(atleastgiventheamount
Comparingfittedandunfittedmodelsintermsoftheirprediction oftrainingdataused).
ofdissimilaritiesprovidesadditionalevidenceforinterpretingthe The method of mixed RSA, which we use here, compares
results. When the unfitted model outperforms the fitted model, representations between models and brain areas at the level of

188 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
|     |     |     |     |     |     | images see | supplementary |     | methods |     | (see Appendix |     | A) or refer |
| --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | ------- | --- | ------------- | --- | ----------- |
toHenriksson,Khaligh-Razavi,Kay,andKriegeskorte(2015),Kay
etal.(2008).
|     |     |     |     |     |     | The RDM | correlation |     | between | brains | ROIs | and | models is |
| --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | ------- | ------ | ---- | --- | --------- |
computedbasedonthe120testingstimuli.ForeachbrainROI,we
×
|     |     |     |     |     |     | had ten | 12 12 | RDMs, | one for | each | experimental |     | run (10 runs |
| --- | --- | --- | --- | --- | --- | ------- | ----- | ----- | ------- | ---- | ------------ | --- | ------------ |
with12differentimagesineach=120distinctimagesoverall).
|     |     |     |     |     |     | Each test       | image   | was presented |       | 13 times | per         | run. | To calculate |
| --- | --- | --- | --- | --- | --- | --------------- | ------- | ------------- | ----- | -------- | ----------- | ---- | ------------ |
|     |     |     |     |     |     | the correlation | between |               | model | and      | brain RDMs, |      | within each  |
experimentalrun,alltrialswereaveraged,yieldingone12×12
RDMforeachrun.Thereportedmodel-to-brainRDMcorrelations
aretheaverageRDMcorrelationsforthetensetsof12images.
|     |     |     |     |     |     | To judge | the ability | of  | a model | RDM | to explain | a   | brain RDM, |
| --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | ------- | --- | ---------- | --- | ---------- |
τ
|     |     |     |     |     |     | we used    | Kendall’s | rank      | correlation | coefficient |              | A (which | is the     |
| --- | --- | --- | --- | --- | --- | ---------- | --------- | --------- | ----------- | ----------- | ------------ | -------- | ---------- |
|     |     |     |     |     |     | proportion | of pairs  | of values |             | that are    | consistently |          | ordered in |
bothvariables).Whencomparingmodelsthatpredicttiedranks
|     |     |     |     |     |     | (e.g. category | model | RDMs) | to  | models | that make | more | detailed |
| --- | --- | --- | --- | --- | --- | -------------- | ----- | ----- | --- | ------ | --------- | ---- | -------- |
predictions(e.g.brainRDMs,object-visionmodelRDMs)Kendall’s
τ
|     |     |     |     |     |     | correlation | is  | recommended |     | (Nili | et al., | 2014), | because the |
| --- | --- | --- | --- | --- | --- | ----------- | --- | ----------- | --- | ----- | ------- | ------ | ----------- |
A
|     |     |     |     |     |     | Pearson   | and Spearman |       | correlation | coefficients |      | have  | a tendency  |
| --- | --- | --- | --- | --- | --- | --------- | ------------ | ----- | ----------- | ------------ | ---- | ----- | ----------- |
|     |     |     |     |     |     | to prefer | a simplified | model | that        | predicts     | tied | ranks | for similar |
dissimilaritiesoverthetruemodel.
Inter-subjectbrainRDMcorrelations:Theinter-subjectbrain
|     |     |     |     |     |     | RDM correlation |     | is computed | for | each | ROI for | comparison | with |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ----------- | --- | ---- | ------- | ---------- | ---- |
model-to-brainRDMcorrelations.Thismeasureisdefinedasthe
τ correlationoftheten12×12RDMs(120test
|     |     |     |     |     |     | averageKendall |     | A   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
stimuli)betweenthetwosubjects(Figs.4–7).
|     |     |     |     |     |     | 2.3. Models |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Fig.3. FixedversusmixedRSA.(A)FixedRSA:abrainRDMiscomparedtoa
modelRDMwhichisconstructedfromthemodelfeatures.Themodelfeatures We tested a total of 20 unsupervised computational model
areextractedfromtheimagesandtheRDMisconstructedusingthepairwise representations,aswellasdifferentlayersofapre-traineddeep
dissimilaritiesbetweenthemodelfeatures.(B)MixedRSA:abrainRDMiscompared
|     |     |     |     |     |     | supervised | convolutional |     | neuronal | network |     | (Krizhevsky | et al., |
| --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | -------- | ------- | --- | ----------- | ------- |
toamodelRDMwhichisconstructedfrommixedmodelfeaturesobtainedvia
|     |     |     |     |     |     | 2012). In | this context, | by  | unsupervised, |     | we  | mean | object-vision |
| --- | --- | --- | --- | --- | --- | --------- | ------------- | --- | ------------- | --- | --- | ---- | ------------- |
receptivefieldmodelling(seealsoFig.2).Thereisfirstatrainingphaseinwhichthe
receptivefieldmodelsareestimatedforeachvoxel(similartoKayetal.(2008)).In modelsthathadnotrainingphase(e.g.featureextractors,suchas
thetestingphase,usingthelearnedreceptivefieldmodels,voxelresponsestonew gist),aswellasmodelsthataretrainedbutwithoutusingimage
stimuliarepredicted.ThentheRDMofthepredictedvoxelresponsesiscompared
labels(e.g.HMAXmodeltrainedwithsomenaturalimages).Some
withtheRDMoftheactualmeasuredbrain(voxel)responses.
ofthemodelsmimicthestructureoftheventralvisualpathway
|     |     |     |     |     |     | (e.g. V1 | model, HMAX); |     | others | are | more broadly |     | biologically |
| --- | --- | --- | --- | --- | --- | -------- | ------------- | --- | ------ | --- | ------------ | --- | ------------ |
representational dissimilarities. This enables direct testing of motivated (e.g. BioTransform, convolutional networks); and the
unfittedmodelsandstraightforwardcomparisonsbetweenfitted others are well-known computer-vision models (e.g. GIST, SIFT,
and unfitted models. The same conceptual question could be PHOG,self-similarityfeatures,geometricblur).Someofthemodels
addressed in the framework of voxel-RF modelling. This would use features constructed by engineers without training with
require fitting linear models to the voxels with the constraint naturalimages(e.g.GIST,SIFT,PHOG).Othersweretrainedinan
thattheresultingrepresentationalspacespannedbythepredicted unsupervised(e.g.HMAX)orsupervised(deepCNN)fashion.
voxelresponsesreproducestherepresentationaldissimilaritiesof Inthefollowingsectionswefirstcomparetherepresentational
themodel’soriginalfeaturespaceascloselyaspossible. geometry of several unsupervised models with that of early to
|     |     |     |     |     |     | intermediate | and     | higher | visual | areas  | using both | fixed         | RSA and |
| --- | --- | --- | --- | --- | --- | ------------ | ------- | ------ | ------ | ------ | ---------- | ------------- | ------- |
|     |     |     |     |     |     | mixed RSA.   | We will | then   | test   | a deep | supervised | convolutional |         |
2.2. Stimuli,responsemeasurements,andRDMcomputation
networkintermsofitsabilityinexplainingthehierarchyofvision.
Furthermethodologicaldetailsareexplainedinthesupplemen-
| In this study | we used | the experimental | stimuli |     | and fMRI |     |     |     |     |     |     |     |     |
| ------------- | ------- | ---------------- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
tarymaterials(Supplementarymethods,AppendixA).
| data from                                                 | Kay et al. (2008); | also used        | in Güçlü | and     | Gerven |            |     |     |     |     |     |     |     |
| --------------------------------------------------------- | ------------------ | ---------------- | -------- | ------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- |
| (2015); Naselaris,                                        | Prenger,           | Kay, Oliver, and | Gallant  | (2009). | The    |            |     |     |     |     |     |     |     |
| stimuliweregrey-scalenaturalimages.Thetrainingstimuliwere |                    |                  |          |         |        | 3. Results |     |     |     |     |     |     |     |
presentedtosubjectsin5scanningsessionswith5runsineach
session(overall25 experimentalruns). Eachrunconsistedof 70 3.1. EarlyvisualareasexplainedbyGaborwaveletpyramid
distinctimagespresentedtwotimeseach.Thetestingstimuliwere
120 grey-scale natural images. The data for testing stimuli were The Gabor wavelet pyramid (GWP) model was used in Kay
collectedin2scanningsessionswith5runsineachsession(overall et al. (2008) to predict responses of voxels in early visual areas
10 experimental runs). Each run consisted of 12 distinct images in humans. Gabor wavelets are directly related to Gabor filters,
presented13timeseach. sincetheycanbedesignedfordifferentscalesandrotations.The
Wehadearlyvisualareas(i.e.V1,V2),intermediatelevelvisual aimofGWPhasbeentomodelearlystagesofvisualinformation
areas(V3,V4),andLOasoneofthehighervisualareas.TheRDMs processing,andithasbeenshownthat2DGaborfilterscanprovide
foreachROIwerecalculatedbasedon120teststimulipresented agoodfittothereceptivefieldweightfunctionsfoundinsimple
to the subjects. For more information about the data set, and cellsofcatstriatecortex(Jones&Palmer,1987).

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 189
Fig.4. RDMcorrelationofunsupervisedmodelsandanimacymodelwithearlyvisualareas.Barsshowtheaverageoften12×12RDMcorrelations(120teststimuliintotal)
withV1,andV2brainRDMs.Therearetwobarsforeachmodel.Thefirstbar,‘model(notfitted)’,showstheRDMcorrelationofamodelwithabrainROIwithoutfitting
themodelresponsestobrainvoxels(fixedRSA).Thesecondbar(voxelRF-fitted)showstheRDMcorrelationofamodelthatisfittedtothevoxelsofthereferencebrain
ROIusing1750trainingimages(mixedRSA;refertoFigs.1and2toseehowthefittingisdone).Starsaboveeachbarshowstatisticalsignificanceobtainedbysignranktest
(FDRcorrectedat0.05).Smallblackhorizontalbarsshowthatthedifferencebetweenthebarsforamodelisstatisticallysignificant(signranktest,5%significancelevel—not
correctedformultiplecomparisons.ForFDRcorrectedcomparisonseethestatisticalsignificancematricesontheright).Theresultsaretheaverageoverthetwosubjects.
ThegreyhorizontallineforeachROIindicatestheinter-subjectbrainRDMcorrelation.ThisisdefinedastheaverageKendall-tau-acorrelationoftheten12×12RDMs(120
teststimuli)betweenthetwosubjects.Theanimacymodeliscategorical,consistingofasinglebinaryvariable,thereforemixinghasnoeffectonthepredictedRDMrank
order.Wethereforeonlyshowtheunfittedanimacymodel.Thecolour-codedstatisticalsignificancematricesattherightsideofthebargraphsshowwhetheranyofthe
twomodelsperformsignificantlydifferentlyinexplainingthecorrespondingreferencebrainROI(FDRcorrectedat0.05).Modelsareshownbytheircorrespondingnumber;
therearetworows/columnsforeachmodel,thefirstonerepresentsthenot-fittedversionandthesecondonethevoxelRF-fitted.Agreysquareinthematrixshows
thatthecorrespondingmodelsperformsignificantlydifferentlyinexplainingthereferencebrainROI(oneofthemsignificantlyexplainsthereferencebrainROIbetter/
worse).
TheGWPmodelhadthehighestRDMcorrelationwithbothV1, comparisonsbetweenallmodels.Thestatisticalcomparisonsare
andV2(Fig.4).AsforV1,theGWPmodel(voxel-RFfitted)performs basedontwo-sidedsigned-ranktest,FDRcorrectedat0.05).
significantlybetterthanallothermodelsinexplainingV1(seethe The GWP model comes very close to the inter-subject RDM
‘statisticalsignificancematrix’).Similarly,inV2,theGWPmodel correlationofthesetwoearlyvisualareas(V1,andV2),although
(unfitted)performswellinexplainingthisROI.AlthoughGWPhas itdoesnotreachit.Indeed,theinter-subjectRDMcorrelationfor
thehighestcorrelationwithV2,thecorrelationisnotsignificantly thesetwoareas(V1andV2)ismuchhigherthanthosecalculated
higherthanthatoftheV1model,HMAX-C2,andHMAX-C3(the fortheotherareas(seetheinter-subjectRDMcorrelationforV3,
‘statisticalsignificancematrices’inFig.4showpairwisestatistical V4, and LO in Figs. 5 and 6). The highest correlation obtained

190 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
Fig.5. RDMcorrelationofunsupervisedmodelsandanimacymodelwithintermediate-levelvisualareas.Barsshowtheaverageoften12×12RDMcorrelations(120test
stimuliintotal)withV3,andV4brainRDMs.Therearetwobarsforeachmodel.Thefirstbar,‘model(notfitted)’,showstheRDMcorrelationofamodelwithabrainROI
withoutfittingthemodelresponsestobrainvoxels(fixedRSA).Thesecondbar(voxelRF-fitted)showstheRDMcorrelationsofamodelthatisfittedtothevoxelsofthe
referencebrainROI,using1750trainingimages(mixedRSA;refertoFigs.1and2toseehowthefittingisdone).Thegreyhorizontallineineachpanelindicatesinter-subject
RDMcorrelationforthatROI.Thecolour-codedstatisticalsignificancematricesattherightsideofthebargraphsshowwhetheranyofthetwomodelsperformsignificantly
differentlyinexplainingthecorrespondingreferencebrainROI(FDRcorrectedat0.05).ThestatisticalanalysesandconventionshereareanalogoustoFig.4.
betweenamodelandabrainROIisfortheGWPmodelandthe V4.However,notethattheinter-subjectRDMcorrelationislower
earlyvisualareasV1andV2.Thissuggeststhatearlyvisionisbetter inV4comparedtoV1,V2,andV3(Fig.5).
modelledorbetterunderstood,comparedtootherbrainROIs.Itis Intermediate layers of the HMAX model (e.g. C2—model #15
possiblethatthenewerGabor-basedmodelsofearlyvisualareas in Fig. 5) seem to perform slightly better than other models
(Kayetal.,2013)explainearlyvisualareasevenbetter. in explaining intermediate visual areas (Fig. 5)—significantly
ThenextbestmodelinexplainingtheearlyvisualareaV1was better than most of the other unsupervised models (see the
thevoxelRF-fittedgistmodel.ForV2,inadditiontotheGWP,the ‘statisticalsignificancematrices’inFig.5;row/column#15refers
HMAX-C2 and C3 features also showed a high RDM correlation. to the statistical comparison of HMAX-C2 features with other
Overallresultssuggestthatshallowmodelsaregoodinexplaining models—two-sidedsigned-ranktest,FDRcorrectedat0.05).More
early visual areas. Interestingly, all the mentioned models that specifically,forV3,inadditiontotheHMAXC1,C2andC3features,
betterexplainedV1andV2arebuiltbasedonGabor-likefeatures. GWP, V1model (whichis acombination of simpleand complex
cells), gist, and bio-transform also perform similarly well (not
3.2. VisualareasV3andV4explainedbyunsupervisedmodels significantlydifferentfromHMAX-C2).
InV4,thevoxelresponsesseemnoisier(theinter-subjectRDM
Several models show high correlations with V3, and V4, and correlation is lower); and the RDM correlation of models with
someofthemcomeclosetotheinter-subjectRDMcorrelationfor this brain ROI is generally lower. The HMAX-C2 is still among

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 191
Fig.6. RDMcorrelationofunsupervisedmodelsandanimacymodelwithhighervisualareaLO.Barsshowtheaverageoften12×12RDMcorrelations(120teststimuli
intotal)withtheLORDM.Therearetwobarsforeachmodel.Thefirstbar,‘model(notfitted)’,showstheRDMcorrelationofamodelwithabrainROIwithoutfittingthe
modelresponsestobrainvoxels(fixedRSA).Thesecondbar(voxelRF-fitted)showstheRDMcorrelationsofamodelthatisfittedtothevoxelsofthereferencebrainROI,
using1750trainingimages(mixedRSA;refertoFigs.1and2toseehowthefittingisdone).Thegreyhorizontallineindicatestheinter-subjectRDMcorrelationinLO.The
colour-codedstatisticalsignificancematrixattherightsideofthebargraphshowswhetheranyofthetwomodelsperformsignificantlydifferentlyinexplainingLO(FDR
correctedat0.05).ThestatisticalanalysesandconventionshereareanalogoustoFig.4.
the best models that explain V4 significantly better than most supervised convolutional neural network, composed of 8 layers:
oftheotherunsupervisedmodels.Thefollowingmodelsperform 5 convolutional layers, followed by 3 fully connected layers. We
similarlywell(notsignificantlydifferentfromHMAXfeatures)in compared the representational geometry of layers of this model
explainingV4:GWP,gist,V1model,bio-transform,andgssim(for withthatofvisualareasalongthevisualhierarchy(Fig.7).
pairwisestatisticalcomparisonbetweenmodels,seethestatistical Amongallmodels,theonesthatbestexplainLOarethemixed
significancematrixforV4). versionsoflayers6and7ofthedeepconvolutionalnetwork.These
Overall from these results we may conclude that the Gabor- layersalsohaveahighanimate/inanimatecategorizationaccuracy
based models (e.g. GWP, gist, V1 model, and HMAX) provide a (Fig.8(B))—slightlyhigherthanotherlayersofthenetwork.Layer
goodbasisforpredictingvoxelresponsesinthebrainfromearly 6ofthedeepnetcomesclosetotheinter-subjectRDMcorrelation
visualareastointermediatelevels.Moregenerally,intermediate forLOasdoestheanimacymodel.Themixedversionofsomeother
visualareasarebestaccountedforbytheunfittedversionsofthe layersofthedeepconvolutionalnetworkalsocomeclosetotheLO
unsupervised models. It seems that for most of the models the inter-subjectRDMcorrelation(Layers3,4,5,and8),asopposed
mixing does not improve the RDM correlation of unsupervised totheunfittedversions.Remarkably,themixedversionofLayer7
modelfeatureswithearlyandintermediatevisualareas. istheonlymodelthatreachestheinter-subjectRDMcorrelation
forLO.
3.3. Highervisualareasexplainedbymixeddeepsupervisedneural Mixingbringsconsistentbenefitstothedeepsupervisedneu-
netlayers ralnetrepresentations(acrosslayersandvisualareas),butnotto
the shallow unsupervised models. For the deep supervised neu-
For the higher visual area LO (Grill-Spector, Kourtzi, & Kan- ralnetlayers(Fig.7),themixedversionspredictbrainrepresen-
wisher,2001;Mack,Preston,&Love,2013),afewoftheunsuper- tations significantly better than the unmixed versions in 85% of
visedmodelsexplainedasignificantamountofnon-noisevariance
thecases(34of40inferentialcomparisons;8layers∗5regions =
(Fig.6).ThesewereGWP,gist,geometricblur(GB),ssim,andbio- 40comparisons).Forshallowunsupervisedmodels(Figs.4–6),by
transform (1st stage). None of these models reached the inter- contrast, the mixed versions predict significantly better in only
subject RDM correlation for LO. Animacy model achieved the 2% of the cases (2 of 100 comparisons; gist with V1, GWP with
highest RDM correlation (though not significantly higher than LO).Forallother98comparisons(98%)betweenmixedandfixed
someoftheotherunsupervisedmodels).Theanimacymodelisa unsupervisedmodels,the fixedmodelseitherperformthe same
simplemodelRDMthatshowstheanimate–inanimatedistinction (e.g.ssim,LBP,SIFT)orsignificantlybetterthanthemixedversions
(itisnotanimage-computablemodel).Theanimacycamecloseto (e.g.HMAX,V1model).
theinter-subjectRDMcorrelationforLO,butdidnotreachit. To assess the ability of object-vision models in the ani-
In 2012, a deep supervised convolutional neural network mate/inanimatecategorizationtask,wetrainedalinearSVMclas-
trained with 1.2 million labelled images (Krizhevsky et al., sifierforeachmodelusingthemodelfeaturesextractedfrom1750
2012) won the ImageNet competition (Deng et al., 2009) at training images (Fig. 8). Animacy is strongly reflected in human
1000-category classification. It achieved top-1 and top-5 error andmonkeyhigherventral-streamareas(Kiani,Esteky,Mirpour,
rates on the ImageNet data that was significantly better than & Tanaka, 2007; Kriegeskorte et al., 2008a; Naselaris, Stansbury,
previous state-of-the-art results on this data set. Following &Gallant,2012).Weusedthe120teststimuliasthetestset.To
Khaligh-Razavi and Kriegeskorte (2014), we tested this deep assesswhethercategorizationaccuracyonthetestsetwasabove

192 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
Fig.7. RDMcorrelationofthedeepsupervisedconvolutionalnetworkwithbrainROIsacrossthevisualhierarchy.Barsshowtheaverageoften12×12RDMcorrelations
(120teststimuliintotal)betweendifferentlayersofthedeepconvolutionalnetworkwitheachofthebrainROIs.Therearetwobarsforeachlayerofthemodel:thefixed
RSA(model-notfitted),andthemixedRSA(voxelRF-fitted).Thegreyhorizontallineineachpanelindicatestheinter-subjectbrainRDMcorrelationforthegivenROI.The
colour-codedstatisticalsignificancematricesshowwhetheranyofthetwomodelsperformsignificantlydifferentlyinexplainingthecorrespondingreferencebrainROI
(FDRcorrectedat0.05).ThestatisticalanalysesandconventionshereareanalogoustoFig.4.
chancelevel, weperformed apermutation test,in whichwe re- issignificantlylowerthanthatofthehigherlayersofthedeepnet-
< .
trainedtheSVMson10,000(category-orthogonalized)randomdi- work(Layers6and7,p 005).
chotomies among the stimuli. Light grey bars in Fig. 8 show the Comparing the animate/inanimate categorization accuracy of
modelcategorizationaccuracyonthe120teststimuli.Categoriza- thelayersofthedeepconvolutionalnetwork(Fig.8(B))withother
tionperformancewassignificantlygreaterthanchanceforfewof models (Fig. 8(A)) showed that the deep convolutional network
theunsupervisedmodels,andallthelayersofthedeepConvNet, is generally better at this task; particularly higher layers of the
exceptLayer1.Interestinglysimplemodels,suchasGWPandgist, modelperformbetter.Incontrasttotheunsupervisedmodels,the
alsoperformabovechanceatthistask,thoughtheirperformance deepconvolutionalnetworkhadbeentrainedwithmanylabelled

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 193
|     |     |     |     |     | geometry         | is already      |               | somewhat     | similar     | to         | that        | of early | and     |
| --- | --- | --- | --- | --- | ---------------- | --------------- | ------------- | ------------ | ----------- | ---------- | ----------- | -------- | ------- |
|     |     |     |     |     | mid-level        | visual          | areas.        | Remixing     |             | is not     | required    | for      | these   |
|     |     |     |     |     | models           | (and associated |               | with         | a moderate  |            | overfitting |          | cost to |
|     |     |     |     |     | generalization   |                 | performance). |              | The benefit | of         | linear      | fitting  | of the  |
|     |     |     |     |     | representational |                 | space         | is therefore |             | outweighed | by          | the      | cost to |
predictionperformanceofoverfitting.Thedeepnet,bycontrast,
|     |     |     |     |     | has features      | optimized |     | to          | distinguish    | a         | set 1000  | categories, |       |
| --- | --- | --- | --- | --- | ----------------- | --------- | --- | ----------- | -------------- | --------- | --------- | ----------- | ----- |
|     |     |     |     |     | whose frequencies |           | are | not matched |                | to either | the       | natural     | world |
|     |     |     |     |     | or the prominence |           | of  | their       | representation |           | in visual | cortex.     | For   |
example,dogspecieswerelikelyoverrepresentedinthetraining
set.Althoughtheresultingsemanticfeaturesarerelatedtothose
emphasizedbytheventralvisualstream,theirrelativeprominence
isincorrectinthemodelrepresentationandfittingisessential.
|     |     |     |     |     | This is         | consistent |        | with our | previous | study  | (Khaligh-Razavi |             |     |
| --- | --- | --- | --- | --- | --------------- | ---------- | ------ | -------- | -------- | ------ | --------------- | ----------- | --- |
|     |     |     |     |     | & Kriegeskorte, |            | 2014), | in which | we       | showed | that            | by remixing |     |
andreweightingfeaturesfromthedeepsupervisedconvolutional
network,wecouldfullyexplaintheITrepresentationalgeometry
foradifferentdataset(thatfromKriegeskorteetal.(2008a)).Note,
however,thatthemethodformixingusedinthatstudyisdifferent
fromtheoneinthismanuscriptasfurtherdiscussedbelowinthe
Discussionunder‘ProsandconsoffixedRSA,voxel-RFmodelling,
andmixedRSA’.
Weknowthatamodel(thevoxel-receptive-fieldmodelhere)
mightnotgeneralizeforacombinationoftworeasons:
(1) Voxel-RFmodelparametersareoverfittedtothetrainingdata.
|     |     |     |     |     | This         | is usually | prevented |      | or reduced | by       | regularization. |      | We     |
| --- | --- | --- | --- | --- | ------------ | ---------- | --------- | ---- | ---------- | -------- | --------------- | ---- | ------ |
|     |     |     |     |     | did gradient |            | descent   | with | early      | stopping | (which          | is a | way of |
regularization)topreventoverfitting.
(2) Themodelfeaturesdonotspanarepresentationalspacethat
|     |     |     |     |     | can explain |     | the brain | representation. |     | This | is the | problem | of  |
| --- | --- | --- | --- | --- | ----------- | --- | --------- | --------------- | --- | ---- | ------ | ------- | --- |
modelmisspecification.Themodelspacedoesnotincludethe
truemodel,orevenagoodmodel.
|     |     |     |     |     | In our   | case             | the lack | of generalization |       | does       | not   | happen     | in the |
| --- | --- | --- | --- | --- | -------- | ---------------- | -------- | ----------------- | ----- | ---------- | ----- | ---------- | ------ |
|     |     |     |     |     | deep net | (in which        | we       | have              | many  | features), | but   | it happens | in     |
|     |     |     |     |     | some of  | the unsupervised |          | models,           | which | have       | fewer | number     | of     |
featuresthanthedeepnet.(Fig.9showsthenumberoffeatures
Fig.8. Animate–inanimatecategorizationperformancefor(A)severalunsuper-
visedmodelsand(B)layersofadeepconvolutionalnetwork.Barsshowanimate for each model.) The fact that fitting brings greater benefits to
vs.inanimatecategorizationperformanceforeachofthemodelsshownonthex-
generalizationperformanceforthemodelswithmoreparameters
axis.AlinearSVMclassifierwastrainedusing1750trainingimagesandtestedby
|     |     |     |     |     | is inconsistent |     | with the | overfitting |     | account. | Instead | we  | suspect |
| --- | --- | --- | --- | --- | --------------- | --- | -------- | ----------- | --- | -------- | ------- | --- | ------- |
120testimages.Pvaluesthatareshownbyasterisksshowwhetherthecatego-
rizationperformancessignificantlydifferfromchance[p < 0.05:*,p < 0.01:**, that the unsupervised models are missing essential nonlinear
p<0.001:***].Pvalueswereobtainedbyrandompermutationofthelabels(num- featuresneededtoexplainhigherventral-streamareaLO.
berofpermutations=10,000).
|         |                    |             |            |              | 3.5. Early | layers | of the | deep | convolutional | network |     | are inferior | to  |
| ------- | ------------------ | ----------- | ---------- | ------------ | ---------- | ------ | ------ | ---- | ------------- | ------- | --- | ------------ | --- |
| images. | Animacy is clearly | represented | in both LO | and the deep |            |        |        |      |               |         |     |              |     |
GWPinexplainingtheearlyvisualareas
| net’s higher | layers. Note, | however, | that the idealized | animacy |     |     |     |     |     |     |     |     |     |
| ------------ | ------------- | -------- | ------------------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
RDMdidnotreachtheinter-subjectRDMcorrelationforLO.Only
the deep net’s Layer 7 (remixed) reached the inter-subject RDM Althoughthehigherlayersofthedeepconvolutionalnetwork
correlation. successfully work as the best model in explaining higher visual
|     |     |     |     |     | areas, the | early | layers | of the | model | are | not as | successful | in  |
| --- | --- | --- | --- | --- | ---------- | ----- | ------ | ------ | ----- | --- | ------ | ---------- | --- |
3.4. Whydoesmixinghelpthesupervisedmodelfeatures,butnotthe explaining the early visual areas. The early visual areas (V1 and
unsupervisedmodelfeatures? V2) are best explained by GWP model. The best layers of the
|     |     |     |     |     | deep convolutional |     | network |     | are ranked | as  | the 4th | best | model |
| --- | --- | --- | --- | --- | ------------------ | --- | ------- | --- | ---------- | --- | ------- | ---- | ----- |
Overall, fitting linear recombinations of the supervised deep in explaining V1, and the 6th best model in explaining V2. The
net’s features gave significantly higher RDM correlations with RDMcorrelationsofthefirsttwolayersofthedeepconvolutional
brain ROIs than using the unfitted deep net representations network with V1 are 0.185 (Layer 1; voxel RF-fitted) and 0.18
(Fig.7).Theoppositetendedtoholdfortheunsupervisedmodels (Layer 2; voxel RF-fitted), respectively. On the other hand, the
(Figs.4–6).Forexample,themixedfeaturesforall8layersofthe RDM correlation of the GWP model (voxel RF-fitted) with V1 is
supervised deep net have significantly higher RDM correlations 0.3, which is significantly higher than that of the early layers of
|     |     |     |     |     |     |     |     | < . |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withLOthantheunmixedfeatures(Fig.7).Bycontrast,onlyone the deep convNet (p 0001, signed-rank test). GWP appears
of the unsupervised models (GWP) better explains LO when its to provide a better account of the early visual system than the
featuresaremixed(Fig.6). earlylayersofthedeepconvolutionalnetwork.Thissuggeststhe
Whydofeaturesfromthedeepconvolutionalnetworkrequire possibilitythatimprovingthefeaturesinearlylayersofthedeep
remixing, whereas the unsupervised features do not? One convolutionalnetwork,inawaythatmakesthemmoresimilarto
interpretationisthattheunsupervisedfeaturesprovidegeneral- humanearlyvisualareas,mightimprovetheperformanceofthe
| purposerepresentationsofnaturalimageswhoserepresentational |     |     |     |     | model. |     |     |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |

194 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
|     |     |     |     |     |     | assessment,    | the | brain-to-model |     | similarity |     | is measured |              | at the |
| --- | --- | --- | --- | --- | --- | -------------- | --- | -------------- | --- | ---------- | --- | ----------- | ------------ | ------ |
|     |     |     |     |     |     | level response |     | patterns.      | We  | assessed   | the | statistical | significance |        |
ofmodel-to-braincorrelationsusingtwo-sidedsigned-ranktest,
correctedformultiplecomparisonsusingFDRat0.05.Modelsthat
|     |     |     |     |     |     | did not | significantly | predict |     | the representation |     | in  | a given | brain |
| --- | --- | --- | --- | --- | --- | ------- | ------------- | ------- | --- | ------------------ | --- | --- | ------- | ----- |
regionfallwithinthetransparentgreyarea(Fig.10).
|     |     |     |     |     |     | The consistency |     | between |     | two approaches |     | was defined |     | as the |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ------- | --- | -------------- | --- | ----------- | --- | ------ |
Spearmanrankcorrelationbetweenthetwosetsofmodel-to-brain
|     |     |     |     |     |     | similarities. | The  | model-to-brain |           | similarities |     | measured  | by            | these |
| --- | --- | --- | --- | --- | --- | ------------- | ---- | -------------- | --------- | ------------ | --- | --------- | ------------- | ----- |
|     |     |     |     |     |     | two methods   | were | highly         | (Spearman |              | r > | 07) . and | significantly |       |
correlatedforallbrainareas.Inotherwords,thetwoapproaches
|     |     |     |     |     |     | gave highly | consistent |     | results. | Consistency |                 | was | even          | higher |
| --- | --- | --- | --- | --- | --- | ----------- | ---------- | --- | -------- | ----------- | --------------- | --- | ------------- | ------ |
|     |     |     |     |     |     | for models  | explaining |     | the      | brain       | representations |     | significantly |        |
(Fig.10).
|     |     |     |     |     |     | When | a model | fails | to explain | the | brain | data, | comparing | its |
| --- | --- | --- | --- | --- | --- | ---- | ------- | ----- | ---------- | --- | ----- | ----- | --------- | --- |
mixedtoitsfixedperformancewithRSAprovidesausefulcluefor
|     |     |     |     |     |     | interpretation |     | of the model’s |     | potential. | Superior | performance |     | of  |
| --- | --- | --- | --- | --- | --- | -------------- | --- | -------------- | --- | ---------- | -------- | ----------- | --- | --- |
thefixedmodelwouldindicateoverfittingofthevoxel-RFmodel.
Failureofbothmixedandfixedversionsofamodelsuggeststhat
thenonlinearfeaturesprovidedbythemodelarenotappropriate
(orthatmixingisrequired,buttheestimateisoverfitteddueto
insufficienttrainingdata)(Fig.11).
|     |     |     |     |     |     | For example,  |             | HMAX-C2 | (model | number    |      | 15 in Fig. | 10)   | has an |
| --- | --- | --- | --- | --- | --- | ------------- | ----------- | ------- | ------ | --------- | ---- | ---------- | ----- | ------ |
|     |     |     |     |     |     | insignificant | correlation |         | with   | all brain | ROIs | in both    | mixed | RSA    |
andvoxel-RFmodelling.However,Figs.3and4showthatwithout
mixingthefixedHMAX-C2versionexplainsbrainrepresentations
significantlybetter.TherawHMAX-C2featuresexplainsignificant
varianceintheV1,V2,V3,andV4representations.Thiswouldhave
Fig.9. Numberoffeaturespermodel.Thehorizontalbarsshowpercentageofmax
goneunnoticedifanalysishadreliedonlyonthevoxel-RFmodel
numberofmodelfeatures.Theabsolutenumberofmodelfeaturesiswrittenin
predictions.
bracketsinfrontofeachmodel.Thenameforeachmodeliswritteninfrontofits
Insummary,mixedRSAenablesustocomparethepredictive
bar.
|     |     |     |     |     |     | performance | of  | the mixed | and | the | fixed | model. | This provides |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --------- | --- | --- | ----- | ------ | ------------- | --- |
usefulinformationabout(1)thepredictiveperformanceoftheraw
| Güçlü | and van Gerven | (2014) | showed | that shallow | features |                        |     |     |       |         |     |          |               |     |
| ----- | -------------- | ------ | ------ | ------------ | -------- | ---------------------- | --- | --- | ----- | ------- | --- | -------- | ------------- | --- |
|       |                |        |        |              |          | model representational |     |     | space | and (2) | the | benefits | to predictive |     |
learnedfromnaturalimageswithanunsupervisedtechniqueout-
performanceoffittingalinearmixingsoastobestexplainabrain
performedaGaborwaveletmodel.Incontrasttothepresentre-
representation(seeFig.11forasummary).
sults,themodelwastrainedwithoutsupervision,thearchitectures
ofthemodelsandthemethodforcomparingperformancewere
| different. | In another study | (Güçlü | & Gerven, | 2015), | they found | 4. Discussion |     |     |     |     |     |     |     |     |
| ---------- | ---------------- | ------ | --------- | ------ | ---------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
thatacombinationofCNNfeatures(includingfeaturesfromlower
| layers and | higher layers) | explains | early visual | areas | better than |     |     |     |     |     |     |     |     |     |
| ---------- | -------------- | -------- | ------------ | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Visualareaspresentadifficultchallengetocomputationalmod-
GWP.Thisisalsodifferentfromouranalysisherewherewedonot
elling.Weneedtounderstandthetransformationofrepresenta-
combineCNNfeaturesfromdifferentlayers.Futurestudiesshould tionsacrossthestagesofthevisualhierarchy.Hereweinvestigated
determinehowearlyvisualrepresentationscanbeaccountedfor thehierarchyofvisualcortexbycomparingtherepresentational
byshallowrepresentationsobtainedby(1)predefinition,(2)un- geometries of several visual areas (V1-4, LO) with a wide range
supervisedlearning,and(3)supervisedlearningonvarioustasks ofobject-visionmodels,rangingfromunsupervisedtosupervised,
withinthesamearchitecture.
andfromshallowtodeepmodelsTheshallowunsupervisedmod-
elsexplainedrepresentationsinearlyvisualareas;andthedeep
3.6. Mixed RSA and voxel receptive-field modelling yield highly supervisedrepresentationsexplainedhighervisualareas.
consistentestimatesofmodelpredictiveperformance We presented a new method for testing models, mixed RSA,
whichbridgesthegapbetweenRSAandvoxel-RFmodelling.RSA
We quantitatively compared the two methods for assessing and voxel-RF modelling have been used in separate studies for
modelperformancesatexplainingbrainrepresentations(Fig.10). comparingcomputationalmodels(Kayetal.,2008,2013;Khaligh-
Thebrain-to-modelsimilaritywasmeasuredforallmodelsusing Razavi&Kriegeskorte,2014;Kriegeskorteetal.,2008a;Nilietal.,
bothmixedRSAandvoxel-RFmodelling.FormixedRSA,thebrain- 2014),buthadnotbeeneitherdirectlycomparedorintegrated.Our
to-modelsimilaritywasdefinedastheKendalltau-acorrelation directcomparisonheresuggestshighlyconsistentresultsbetween
between model RDMs and brain RDMs on the 120 test stimuli. mixed RSA and voxel-RF modelling—reflecting the fact that the
For voxel RF, the brain-to-model similarity was defined as the same method is used to fit a linear transform using the training
Kendalltau-acorrelationbetweenpredictedvoxelresponsesand data. The difference lies in the level at which model predictions
the actual voxel responses. For each test stimulus, its predicted are compared to data: the level of brain responses in voxel-RF
voxel responses are correlated with measured voxel responses, modellingandthelevelofrepresentationaldissimilaritiesinRSA.
giving us 120 correlation values (for 120 test stimuli), which Wealsoshowedthatalinearlymixedmodelcanperformbetter
were then averaged. These values are shown for each model (benefittingfromfitting)orworse(sufferingfromoverfitting)than
and each ROI in Fig. 10 (values on the x-axis). In other words, thefixedoriginalmodelrepresentation.Comparingthepredictive
in mixed RSA, the brain-to-model similarity is measured at the performanceofmixedandfixedversionsofeachmodelprovides
level of representational dissimilarities; and in voxel-RF model usefulconstraintsforinterpretation.

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 195
Fig.10. MixedRSAandvoxel-RFmodellingarehighlyconsistent.EachpanelshowstheconsistencybetweenmixedRSAandvoxelRFinassessingmodel-to-brainsimilarity
plottedforeachROIseparately.Eachdotisonemodel(reddots=deepCNNlayers;blackdots=unsupervisedmodels).Numbersindicatethemodel(seeFig.9formodel
numbering).ThesimilaritymeasurewithbrainROIsisKendalltau-acorrelation.FormixedRSA,itistheTau-acorrelationbetweentheRDMofpredictedvoxelresponses
andabrainRDM(comparisonatthelevelofdissimilaritypatterns).ForvoxelRF,however,itistheTau-acorrelationbetweenpredictedvoxelresponsesandactualvoxel
responsesatthelevelofpatternsthemselves(averagedover120teststimuli).Thetransparenthorizontalandverticalrectanglescovernon-significantrangesalongeachaxis
(non-significantbrain-to-modelcorrelation).‘sig-corr’istheconsistencybetweenmixedRSAandvoxelRFinassessingthosemodelsthatfallwithinthesignificantrange
(outsidethetransparentrectangles).TheconsistencyismeasuredastheSpearmanrankcorrelationbetweenmixedRSAbrain-to-modelcorrelationsandvoxelRFbrain-to-
modelcorrelations.‘all-corr’istheconsistencybetweenmixedRSAandvoxelRFacrossallmodels.Bluelinesaretheleastsquarefits(usingalldots).(Forinterpretationof
thereferencestocolourinthisfigurelegend,thereaderisreferredtothewebversionofthisarticle.)
4.1. ProsandconsoffixedRSA,voxel-RFmodelling,andmixedRSA assumptionsthatregularizethefit.Thischallengescaleswiththe
numberofmodelfeatures,eachofwhichrequiresaweighttobe
Voxel-RFmodellingpredictsbrainresponsestoasetofstimuli fittedforeachoftheresponsechannels.
asalinearcombinationofnonlinearfeaturesofthestimuli.One An alternative approach is RSA, which compares brain and
challengewiththisapproachistoavoidoverfitting,throughuse modelrepresentationsatthelevelofthedissimilaritystructureof
ofasufficientlylargetrainingdatasetincombinationwithprior theresponsepatterns(Kriegeskorte,2009;Kriegeskorte&Kievit,

196 S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197
theregularizationpenalty)aboutthemixingweights.Theformer
|     |     |     |     |     |     | is costly | and the | latter | affects | the interpretation |     | of the results, |
| --- | --- | --- | --- | --- | --- | --------- | ------- | ------ | ------- | ------------------ | --- | --------------- |
becausethepriorispartofthemodel.
|     |     |     |     |     |     | The fact | that | mixed unsupervised |     | models | tended | to perform |
| --- | --- | --- | --- | --- | --- | -------- | ---- | ------------------ | --- | ------ | ------ | ---------- |
worsethantheirfixedversionsillustratesthatmixingcanentail
overfittingandshouldnotingeneralbeinterpretedastestingthe
bestofallmixedmodels.Ifalargeamountofdataisavailablefor
training,mixedRSAcombinestheflexibilityofthevoxel-RFmodel
fittingwiththestabilityandadditionalinterpretationalconstraint
providedbytestingfixedversionsofthemodelswithRSA.
4.2. Performanceofdifferentmodelsacrossthevisualhierarchy
Themodelsthatwetestedherewereallfeedforwardmodels
Fig.11. MixedRSAvs.FixedRSA.Thefigureshowshowtointerpretresultsfrom
mixedRSAandfixedRSA.IfamodelexplainsbraindatainbothmixedandfixedRSA ofvision,fromshallowunsupervisedfeatureextractors(e.g.SIFT)
(top-left),thissuggeststhatthemodelfeaturesarealreadygoodforexplainingthe to a deep supervised convolutional neural network model
braindata,obviatingtheneedforfeatureremixing.Ifamodelonlyexplainsbrain that can perform object categorization. We explored a wide
datausingfixedRSAbutnotthemixedRSA(top-right),itsuggeststhatthelinear range of models (two model instantiations for each of the
mixingisoverfitted.Ifontheotherhand,amodelonlyexplainsbraindatausing
|     |     |     |     |     |     |     |     |     | +   |     |     | =   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mixedRSAbutnotthefixedRSA(bottom-left),itshowsthatthemodelfeatures 28 model representations the animacy model 57 model
provideagoodbasisforexplainingthebraindataandjustneedtobelinearly representations in total), extending previous findings (Khaligh-
remixed.Finally,ifamodelfailstoexplainbraindatausingbothfixedandmixed
|     |     |     |     |     |     | Razavi & | Kriegeskorte, | 2013, | 2014) | to the | data set | of Kay et al. |
| --- | --- | --- | --- | --- | --- | -------- | ------------- | ----- | ----- | ------ | -------- | ------------- |
RSA(bottom-right),themodelfeaturesmaynotprovideagoodbasisforexplaining
(2008)andtomultiplevisualareas.
thebraindata.
|                    |     |             |         |             |             | 4.2.1. Fixed | shallow | unsupervised |     | models explain |     | the lower-level |
| ------------------ | --- | ----------- | ------- | ----------- | ----------- | ------------ | ------- | ------------ | --- | -------------- | --- | --------------- |
| 2013; Kriegeskorte | et  | al., 2008b; | Nili et | al., 2014). | This method |              |         |              |     |                |     |                 |
representations
| enables us | to test fixed | models | directly | with data | from a single |             |     |              |        |           |             |       |
| ---------- | ------------- | ------ | -------- | --------- | ------------- | ----------- | --- | ------------ | ------ | --------- | ----------- | ----- |
|            |               |        |          |           |               | The shallow |     | unsupervised | models | explained | substantial | vari- |
stimulusset.Sincethemodelisfixed,weneednotworryabout
|     |     |     |     |     |     | ance components |     | of the | early | visual representations, |     | although |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ------ | ----- | ----------------------- | --- | -------- |
overfitting,andnotrainingdataisneeded.However,ifamodelfails
theydidnotreachtheinter-subjectRDMcorrelation.Themixed
toexplainabrainrepresentation,itmaystillbethecasethatits
|     |     |     |     |     |     | versions | of the | unsupervised | models | sometimes |     | performed sig- |
| --- | --- | --- | --- | --- | --- | -------- | ------ | ------------ | ------ | --------- | --- | -------------- |
featuresprovidethenecessarybasis,butrequirelinearremixing.
|     |     |     |     |     |     | nificantly | worse | than the | original | versions | of  | those models |
| --- | --- | --- | --- | --- | --- | ---------- | ----- | -------- | -------- | -------- | --- | ------------ |
InKhaligh-RazaviandKriegeskorte(2014),wehadbraindata
(e.g.HMAX-C2).Forlowervisualareas,thefixedshallowunsuper-
| for a set | of only 96 | stimuli. | We did not | have | a large separate |     |     |     |     |     |     |     |
| --------- | ---------- | -------- | ---------- | ---- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
visedmodelsappeartoalreadyapproximatetherepresentational
| training | set of brain | responses | to different | images | for fitting a |     |     |     |     |     |     |     |
| -------- | ------------ | --------- | ------------ | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- |
linear mix of the large numbers of features of the models we spacesquitewell.
weretesting.Toovercomethisproblem,wefittedlinearcombi- For higher visual areas, the unsupervised models were not
successfuleitherwithorwithoutmixing.Noneoftheunsupervised
| nations of | the features, | so as     | to emphasize   | categorical | divisions       |                 |       |          |        |               |        |              |
| ---------- | ------------- | --------- | -------------- | ----------- | --------------- | --------------- | ----- | -------- | ------ | ------------- | ------ | ------------ |
|            |               |           |                |             |                 | models came     | close | to       | the LO | inter-subject | RDM    | correlation. |
| known to   | be prevalent  | in higher | ventral-stream |             | representations |                 |       |          |        |               |        |              |
|            |               |           |                |             |                 | One explanation |       | for this | is     | that these    | models | are missing  |
(animate/inanimate,face/non-face,body/non-body).Thisrequired
|     |     |     |     |     |     | the visuo-semantic |     | nonlinear | features | needed | to  | explain these |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | --------- | -------- | ------ | --- | ------------- |
categorylabelsforaseparatesetoftrainingimages,butnoaddi-
representations.
tionalbrain-activitydata.Wethencombinedthelinearcategory
readoutswiththemodelfeaturesbyfittingonlyasmallnumberof
prevalenceweights(oneforeachmodellayerandoneforeachof 4.2.2. Mixeddeepsupervisedneuralnetworkexplainsthehigher-level
| the3categoryreadoutdimensions)withnonnegativeleastsquares |     |     |     |     |     | representation |     |     |     |     |     |     |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
(seealso,Jozwiketal.,2015).Thefittingofthesefewparameters The lower layers of the deep supervised network performed
didnotrequirealargetrainingsetcomprisingmanyimages.We slightlyworsethanthebestunsupervisedmodelatexplainingthe
earlyvisualrepresentations.However,itshigherlayersperformed
couldusethesetof96images,avoidingthecircularityofoverfit-
bestatexplainingthehigher-levelLOrepresentation.Importantly,
ting(Kriegeskorte,Simmons,Bellgowan,&Baker,2009)bycross
validationwithinthisset. theonlymodeltoreachtheinter-subjectRDMcorrelationforLO
In the present analyses, we had enough training data to fit was the mixed version of Layer 7 of the deep net. Whereas the
linearcombinationsofmodelfeaturesinordertobestexplainbrain mixedversionsoftheunsupervisedmodelsperformedsimilaror
responses. We first fit a linear model to predict voxel responses worsethanthefixedversions,themixedversionsofthelayersof
with computational models, as in voxel-RF modelling, and then thedeepsupervisednetperformedsignificantlybetterthantheir
constructed RDMs from the predicted responses and compared fixedcounterparts.
them with the brain RDMs, for a separate set of test stimuli. Adeeparchitecturetrainedtoemphasizetherightcategorical
This approach enables us to test both mixed and fixed models, divisionsappearstobeessentialforexplainingthecomputations
combiningthestrengthsofbothapproaches. underlying for the visuo-semantic representations in higher
ventral-streamvisualareas.
Mixingenablesustoinvestigatewhetheralinearcombination
| of model | features can | provide | a better | explanation | of the brain |     |     |     |     |     |     |     |
| -------- | ------------ | ------- | -------- | ----------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
representational geometry. This helps address the question of Acknowledgments
whetherthemodelfeatures(a)provideagoodbasisforexplaining
abrainregionandjustneedtobeappropriatelylinearlycombined WewouldliketothankKatherineStorrsforhelpfulcomments
or (b) the model features do not provide a good basis of the on the manuscript. We would also like to thank all those
brain representation (see Fig. 11 for a comparison between who shared their model implementations with us. In particular
mixed and fixed RSA in this regard). However, mixing requires Pavel Sountsov and John Lisman, who kindly helped us to set
a combination of (1) substantial additional training data for a up their code. This work was supported by the UK Medical
separatesetofstimuliand(2)priorassumptions(e.g.implicitto ResearchCouncil,aCambridgeOverseasTrustandYousefJameel

S.-M.Khaligh-Razavietal./JournalofMathematicalPsychology76(2017)184–197 197
Scholarship to SK; an Aalto University Fellowship Grant and an Kay,K.N.,Naselaris,T.,Prenger,R.J.,&Gallant,J.L.(2008).Identifyingnatural
AcademyofFinlandPostdoctoralResearcherGrant(278957)toLH, imagesfromhumanbrainactivity.Nature,452,352–355.
Kay,K.N.,Winawer,J.,Rokem,A.,Mezer,A.,&Wandell,B.A.(2013).Atwo-stage
andaEuropeanResearchCouncilStartingGrant(261352)toNK.
cascademodelofBOLDresponsesinhumanvisualcortex.PLoSComputational
Theauthorsdeclarenocompetingfinancialinterests.
Biology,9,e1003079.
Khaligh-Razavi,S.-M.,&Kriegeskorte,N.(2013).Object-visionmodelsthatbetter
explainITalsocategorizebetter,butallmodelsfailatboth.CosyneAbstracts,
AppendixA. Supplementarymaterial
SaltLakeCityUSA.
Khaligh-Razavi,S.-M.,&Kriegeskorte,N.(2014).Deepsupervised,butnotunsuper-
Supplementary material related to this article can be found vised,modelsmayexplainITcorticalrepresentation.PLoSComputationalBiol-
onlineathttp://dx.doi.org/10.1016/j.jmp.2016.10.007. ogy,10,e1003915.
Kiani,R.,Esteky,H.,Mirpour,K.,&Tanaka,K.(2007).Objectcategorystructurein
responsepatternsofneuronalpopulationinmonkeyinferiortemporalcortex.
References JournalofNeurophysiology,97,4296–4309.
Kriegeskorte,N.(2009).Relatingpopulation-coderepresentationsbetweenman,
monkey,andcomputationalmodels.FrontiersinNeuroscience,3,363–373.
Bell,A.H.,Hadj-Bouziane,F.,Frihauf,J.B.,Tootell,R.B.H.,&Ungerleider,L.G.(2009).
Kriegeskorte,N.(2015).Deepneuralnetworks:Anewframeworkformodeling
Object representations in the temporal cortex of monkeys and humans as
biologicalvisionandbraininformationprocessing.AnnualReviewofVision
revealedbyfunctionalmagneticresonanceimaging.JournalofNeurophysiology,
101,688–700. Science,1,417–446.
Cadieu,C.F.,Hong,H.,Yamins,D.L.K.,Pinto,N.,Ardila,D.,Solomon,E.A.,etal. Kriegeskorte, N., & Kievit, R. A. (2013). Representational geometry: integrating
(2014).DeepneuralnetworksrivaltherepresentationofprimateITcortexfor cognition,computation,andthebrain.TrendsinCognitiveSciences,17,401–412.
corevisualobjectrecognition.PLOSComputationalBiology,10,e1003963. Kriegeskorte, N., Mur, M., & Bandettini, P. (2008b). Representational similarity
Cowen,A.S.,Chun,M.M.,&Kuhl,B.A.(2014).Neuralportraitsofperception: analysis–connectingthebranchesofsystemsneuroscience.FrontiersinSystem
reconstructingfaceimagesfromevokedbrainactivity.Neuroimage,94,12–22. Neuroscience,2,4.
Deng,J.,Dong,W.,Socher,R.,Li,L.-J.,Li,K.,&Fei-Fei,L.(2009).ImageNet:Alarge- Kriegeskorte,N.,Mur,M.,Ruff,D.A.,Kiani,R.,Bodurka,J.,Esteky,H.,etal.(2008a).
scalehierarchicalimagedatabase.InIEEEconferenceoncomputervisionand Matchingcategoricalobjectrepresentationsininferiortemporalcortexofman
patternrecognition,2009.CVPR2009(pp.248–255). andmonkey.Neuron,60,1126–1141.
Dumoulin,S.O.,&Wandell,B.A.(2008).Populationreceptivefieldestimatesin Kriegeskorte,N.,Simmons,W.K.,Bellgowan,P.S.F.,&Baker,C.I.(2009).Circular
humanvisualcortex.NeuroImage,39,647–660. analysis in systems neuroscience: the dangers of double dipping. Nature
Eichhorn,J.,Sinz,F.,&Bethge,M.(2009).NaturalimagecodinginV1:Howmuch Neuroscience,12,535–540.
useisorientationselectivity?PLoSComputationalBiology,5,e1000336. Krizhevsky,A.,Sutskever,I.,&Hinton,G.E.(2012).ImageNetclassificationwith
Ester,E.F.,Sprague,T.C.,&Serences,J.T.(2015).Parietalandfrontalcortexencode deepconvolutionalneuralnetworks.InF.Pereira,C.J.C.Burges,L.Bottou,&K.
stimulus-specificmnemonicrepresentationsduringvisualworkingmemory. Q.Weinberger(Eds.),Advancesinneuralinformationprocessingsystems,Vol.25
Neuron,87(4),893–905. (pp.1097–1105).CurranAssociates,Inc..
Grill-Spector,K.,Kourtzi,Z.,&Kanwisher,N.(2001).Thelateraloccipitalcomplex LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553),
anditsroleinobjectrecognition.VisionResearch,41(10),1409–1422. 436–444.
Grill-Spector,K.,&Weiner,K.S.(2014).Thefunctionalarchitectureoftheventral Mack,M.L.,Preston,A.R.,&Love,B.C.(2013).Decodingthebrain’salgorithm
temporalcortexanditsroleincategorization.NatureReviewsNeuroscience,15, for categorization from its neural implementation. Current Biology, 23(20),
536–548. 2023–2027.
Güçlü,U.,&Gerven,M.A.J.van(2015).Deepneuralnetworksrevealagradientin Mitchell,T.M.,Shinkareva,S.V.,Carlson,A.,Chang,K.-M.,Malave,V.L.,Mason,R.
thecomplexityofneuralrepresentationsacrosstheventralstream.Journalof A.,etal.(2008).Predictinghumanbrainactivityassociatedwiththemeanings
Neuroscience,35,10005–10014. ofnouns.Science,320,1191–1195.
Güçlü,U.,&vanGerven,M.A.(2014).Unsupervisedfeaturelearningimproves Naselaris,T.,Kay,K.N.,Nishimoto,S.,&Gallant,J.L.(2011).Encodinganddecoding
prediction of human brain activity in response to natural images. PLoS infMRI.Neuroimage,56,400–410.
ComputationalBiology,10,e1003724. Naselaris,T.,Prenger,R.J.,Kay,K.N.,Oliver,M.,&Gallant,J.L.(2009).Bayesian
Haxby,J.V.,Gobbini,M.I.,Furey,M.L.,Ishai,A.,Schouten,J.L.,&Pietrini,P.(2001).
reconstruction of natural images from human brain activity. Neuron, 63,
Distributedandoverlappingrepresentationsoffacesandobjectsinventral
902–915.
temporalcortex.Science,293,2425–2430. Naselaris,T.,Stansbury,D.E.,&Gallant,J.L.(2012).Corticalrepresentationof
Hegdé,J.,&VanEssen,D.C.(2000).Selectivityforcomplexshapesinprimatevisual animateandinanimateobjectsincomplexnaturalscenes.JournalofPhysiology-
areaV2.JournalofNeuroscience,20(5),61–66.
Paris,106,239–249.
Henriksson, L., Khaligh-Razavi, S.-M., Kay, K., & Kriegeskorte, N. (2015). Visual Nili,H.,Wingfield,C.,Walther,A.,Su,L.,Marslen-Wilson,W.,&Kriegeskorte,N.
representationsaredominatedbyintrinsicfluctuationscorrelatedbetween
(2014).Atoolboxforrepresentationalsimilarityanalysis.PLoSComputational
areas.NeuroImage,.
Biology,10,e1003553.
Hung,C.P.,Kreiman,G.,Poggio,T.,&DiCarlo,J.J.(2005).Fastreadoutofobject
Pasupathy,A.,&Connor,C.E.(2002).PopulationcodingofshapeinareaV4.Nature
identityfrommacaqueinferiortemporalcortex.Science,310,863.
Neuroscience,5,1332–1338.
Huth,A.G.,Nishimoto,S.,Vu,A.T.,&Gallant,J.L.(2012).Acontinuoussemantic
Skouras,K.,Goutis,C.,&Bramson,M.J.(1994).Estimationinlinearmodelsusing
spacedescribestherepresentationofthousandsofobjectandactioncategories
gradientdescentwithearlystopping.StatisticsandComputing,4,271–278.
acrossthehumanbrain.Neuron,76,1210–1224.
Sprague,T.C.,&Serences,J.T.(2013).Attentionmodulatesspatialprioritymapsin
Jones,J.P.,&Palmer,L.A.(1987).Anevaluationofthetwo-dimensionalGaborfilter
modelofsimplereceptivefieldsincatstriatecortex.JournalofNeurophysiology, thehumanoccipital,parietalandfrontalcortices.NatureNeuroscience,16(12),
58,1233–1258. 1879–1887.
Yamins,D.L.K.,Hong,H.,Cadieu,C.F.,Solomon,E.A.,Seibert,D.,&DiCarlo,J.J.
Jozwik,K.M.,Kriegeskorte,N.,&Mur,M.(2015).Visualfeaturesassteppingstones
towardsemantics:ExplainingobjectsimilarityinITandperceptionwithnon- (2014).Performance-optimizedhierarchicalmodelspredictneuralresponses
negativeleastsquares.Neuropsychologia,. inhighervisualcortex.ProceedingsoftheNationalAcademyofSciencesofthe
Kanwisher,N.,McDermott,J.,&Chun,M.M.(1997).Thefusiformfacearea:a UnitedStatesofAmerica,111,8619–8624.
moduleinhumanextrastriatecortexspecializedforfaceperception.TheJournal Ziemba, C. M., & Freeman, J. (2015). Representing ‘‘stuff’’ in visual cortex.
ofNeuroscience,17,4302–4311. ProceedingsoftheNationalAcademyofSciences,112(4),942–943.