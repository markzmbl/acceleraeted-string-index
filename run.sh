

case "$4" in
7) script=./benchmark-create7;;
13) script=./benchmark-create13;;
16) script=./benchmark-create16;;
32) script=./benchmark-create32;;
esac

$script -f $1 -s $2 -e $3 -p $4 -fs $5 -bs $6 -ms $7 -b $8 -v -csv;
#./benchmark-query -f $1 -s $2 -e $3 -fs $5 -bs $6 -ms $7;
#python3 ./plot_group_distribution.py -f $1 -s $2 -e $3 -fs $5 -bs $6 -ms $7;