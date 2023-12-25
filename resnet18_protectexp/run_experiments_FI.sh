python3 experiment.py --policy Conv2d_Raw_FI_activation --repeat 5
python3 experiment.py --policy ProtectedConv2d_TMR_FI_activation --repeat 5
python3 experiment.py --policy ProtectedConv2d_ABED_Recomp_FI_activation --repeat 5
python3 experiment.py --policy ProtectedConv2d_ABED_Recomp_FI_activation --threshold 1e-4 --repeat 5
python3 experiment.py --dump
