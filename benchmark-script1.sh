
p=7
b=140000000
fs=32768
bs=32
ms=8
s=4

for f in protein7-exponential
do
    for i in $(seq 1 55)
    do
        e="${i}e-8"
        echo "[RUN] $f $e $s"
        echo 
        ./run.sh $f $s $e $p $fs $bs $ms $b
    done
done
