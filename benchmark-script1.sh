
b=40000000
s=4

#for f in gene16-exponential gene16-normal gene32-exponential gene32-normal protein7-exponential protein7-normal isbn isbn-unique
for f in protein7-exponential
do

    if [[ $f == protein7* ]]
    then
        p=7
        ms=8
    elif [[ $f == isbn* ]]
    then
        p=13
        ms=14
    elif [[ $f == gene16* ]]
    then
        p=16
        ms=17
    elif [[ $f == gene32* ]]
    then
        p=32
        ms=33
    fi
    #for i in 1 $(seq 10 10 100)
    for i in $(seq 70 10 100)
    do
        e="${i}e-8"
        fs=32768 #"${i}e7"
        bs=32 #"${i}e5"
        echo "[RUN] $f $e $s $fs $bs"
        echo 
        ./run.sh $f $s $e $p $fs $bs $ms $b
    done
done