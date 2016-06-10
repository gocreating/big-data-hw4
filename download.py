hadoop fs -get result
echo 'origin_cancelled,predict_cancelled' > result/head
cat result/head result/part-* > result.csv
