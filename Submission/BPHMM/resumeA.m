configNPBayesToolbox;
R = 1;
data = ARSeqData( R );
%add = '../../cis-pd/outputA/';
add = '../../../../rcf-proj3/ck1/cis-pd/outputA_Total/'
listing = dir(add);
listing = {listing.name};
pattern = ".csv";
listing = listing(endsWith(listing, pattern) == 1);
for ii=1:size(listing, 2)
       X = readtable(strcat(add, listing{ii}));
      X = X{:,:};
      X = permute(X, [2,1]);
     data = data.addSeq(X, listing{ii}(1:end-4));
end

task = 1;
modelP = {}; 
algP   = {'Niter', 1000 , 'HMM.doSampleHypers',1,'BP.doSampleMass',1,'BP.doSampleConc',1, 'doSampleFUnique', 0, 'doSampleUniqueZ', 1, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven', 'doAnneal', 'Exp'}; %, 'TimeLimit',36000 #'BP.doSampleMass',1,'BP.doSampleConc',1,
initP  = {'F.nTotal', 1}; %'F.nTotal', 1    , @initBPHMMPrevRun, 'jobID', 1, 'taskID', 38, 'InitFunc', @initBPHMMSeq, 'nSubsetObj', 100
%CH = runBPHMM( data, modelP, {2, task}, algP, initP );
CH = resumeBPHMM({15, task,"saveEvery",10,"printEvery",15}, {'Niter', 1000, 'HMM.doSampleHypers',1,'BP.doSampleMass',1,'BP.doSampleConc',1, 'doSampleFUnique', 0, 'doSampleUniqueZ', 1, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven', 'doAnneal', 'Exp'}, {'jobID', 15, 'taskID', task});
exit
