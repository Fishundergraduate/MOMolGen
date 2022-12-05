import json
import argparse

parser = argparse.ArgumentParser(description="revise setting file")

parser.add_argument('targetDir',help="path to dir")
parser.add_argument('N',help='count',type=int)
args = parser.parse_args()

tDir = args.targetDir
N = args.N

if False:
    with open(tDir+'/input/python_config.json','w') as f:
        #config = json.load(f)
        config = dict()
        config['isLoadTree'] = True
        whereisParetoFile = "./present/pareto.json"
        config['whereisRNNmodelDir'] = "./model3/"
        config['limitTimeHours'] = 23
        config['limitTimeMinutes'] = 0
        config['limitTimeSeconds'] = 0
        config['isUseeToxPred'] = True
        config['proteinName'] = '6lu7.pdbqt' if 5<N<11 else '3zosA_prepared.pdbqt'
        config['randomSeed'] = 333
        config['saThreshold'] = 3.5
        config['//'] = "eToxPred must be located in ligand_design/etoxpred_best_model.joblib"
        json.dump(config,f, indent=4, separators=(',', ': '))

with open(tDir+"/output/allproducts.txt","w") as f:
    f.write('')

with open(tDir+"/output/allLigands.txt","w") as f:
    f.write('')

with open(tDir+"/present/depth.txt","w") as f:
    f.write('')

with open(tDir+"/present/error_output.txt","w") as f:
    f.write('')

with open(tDir+"/present/ligands.txt","w") as f:
    f.write('')

with open(tDir+"/present/output.txt","w") as f:
    f.write('')

with open(tDir+'/present/pareto.json','w+') as f:
    pareto = dict()
    pareto['front'] = []
    pareto['size'] = 0
    pareto['avg'] = []
    pareto['compounds'] = []
    json.dump(pareto,f,indent=4,separators=(',',': '))

with open(tDir+"/present/scores.txt","w") as f:
    f.write('')

with open(tDir+"/present/tree.json","w") as f:
    f.write('')

with open(tDir+"/workspace/cvt_log.txt","w") as f:
    f.write('')

with open(tDir+"/workspace/log_docking.txt","w") as f:
    f.write('')



