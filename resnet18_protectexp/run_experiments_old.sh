python3 experiment.py --policy Conv2d_Raw --repeat 5
python3 experiment.py --policy ProtectedConv2d_TMR --repeat 5
python3 experiment.py --policy ProtectedConv2d_ABED_Recomp --repeat 5
python3 experiment.py --policy ProtectedConv2d_ABED_Recomp --threshold 1e-4 --repeat 5
python3 experiment.py --dump --datasize 500
