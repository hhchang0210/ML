#ommand is `sh run.sh 2018-09-10 2018-09-10`
#pip install tensorflow=1.5
#pip install keras

# generate the data to predict
python ./load.py $1 $2 ../output/data 

# predict with a trained model
./predict.py ../output/data ../output/pred

# make decision
./make_decision.py ../output/pred ../commit/$1_$2.json
