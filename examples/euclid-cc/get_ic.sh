echo "Downloading ICs:"
wget -r -l1 --no-parent -nd -P euclidIC_2048_G1 "https://www.ics.uzh.ch/~aurel/euclidIC_2048/"

echo "Converting to BigFile:"
python ../../tools/convert_from_gadget_1.py "euclidIC_2048_G1/Planck2013-Npart_2048_Box_500-Fiducial" "euclidIC_2048"
