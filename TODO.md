### Running Condition

aa

|Data| target protien|run code| Condition | jobsubmit |
|---|---|---| --- | --- |
|1 to 5 | 3zosA|mcts_ligand.py | 3zosA BaseLine | 1|
|6 to 10 | 6lu7 | mcts_ligand.py | 6LU7 BaseLine | 2| 
|11 to 15 | 3zosA | mcts_ligand_DlogPQED.py | D LogP QED | 4|
|16 to 20 | 3zosA | mcts_ligand.py | blank | 5|
| data_test | 3zosA | mcts_ligand_copy.py | develop | 3|
| data_test_2 | receptor.pdbqt | mcts_ligand.py | Compare with ChemTS | 6|
| 21 | 3zosA |mcts_ligand_docking.py |  only docking | 7<- ERROR|

Newer
|Data| target protien|run code| Condition | jobsubmit |
|---|---|---| --- | --- |
|1 to 5 | 3zosA|mcts_ligand_1.py | Docking | 0|
|6 to 10 | 3zosA | mcts_ligand_2.py | Docking QED | 0| 
|11 to 15 | 3zosA | mcts_ligand_3.py | Docking QED Toxicity | 0|
|16 to 20 | 3zosA | mcts_ligand_4.py | D QED Toxicity Membrane permeability| 0|
|21 to 25 | 3zosA | mcts_ligand_5.py | D QED Toxicity Membrane permeability, Metabolic stability | 0|