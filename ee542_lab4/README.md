# ee542_lab4

## Compilation Instructions

Ensure Hadoop is installed (/usr/local/hadoop).
Ensure Spark is installed (/opt/spark).

### Prepare Gutenberg Input
1. Create HDFS input dir

   ```hdfs dfs -mkdir -p /gutenberg```
2. Get Gutenberg text files

   ```wget https://www.gutenberg.org/files/<num>/<num>-0.txt```  where <num> is an int of a book
3. Upload Gutenberg text files

    ```hdfs dfs -put ~/gutenberg/*.txt /gutenberg/```


### Hadoop
1. Start Hadoop
``` start-dfs.sh``` and ``` start-yarn.sh```
2. Run Hadoop Word Count

```
time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \ -files wordcount_mapper.py,wordcount_reducer.py \ -input /gutenberg \ -output /output_wordcount \ -mapper wordcount_mapper.py \ -reducer wordcount_reducer.py
```
3. Run Hadoop Character Count

```
time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \ -files char_mapper.py,char_reducer.py \ -input /gutenberg \ -output /output_charcount \ -mapper char_mapper.py \ -reducer char_reducer.py
```

4. Run Hadoop Min/max

```
time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \ -files minmax_reducer.py \ -input /output_wordcount \ -output /output_minmax \ -mapper /bin/cat \ -reducer minmax_reducer.py
```

View results of via ```hdfs dfs -cat /<output_name>/part-*``` or ```hdfs dfs -cat /<output_name>/part-* | head -20```

Outputs can then be cleared via command ```hdfs dfs -rm -r -f /<output_name>```

Two-Node testing can be completed by adding worker to ```/usr/local/hadoop/etc/hadoop/workers``` and ```stop-yarn.sh``` and ```stop-dfs.sh``` and then repeating steps 1-4

### Spark
1. Run Word Count

```time /opt/spark/bin/spark-submit \ --master local[*] \ spark_wordcount.py```

2. Run Character Count

```time /opt/spark/bin/spark-submit \ --master local[*] \ spark_charcount.py```

3. Run Min/Max

```time /opt/spark/bin/spark-submit \ --master local[*] \ spark_minmax.py```

View results of via ```hdfs dfs -cat /<output_name>/part-*``` or ```hdfs dfs -cat /<output_name>/part-* | head -20```

Outputs can then be cleared via command ```hdfs dfs -rm -r -f /<output_name>```

Two-Node testing can be completed by adding worker to ```/usr/local/hadoop/etc/hadoop/workers``` and ```stop-yarn.sh``` and ```stop-dfs.sh``` and then repeating steps 1-3, but REPLACE ```--master local[*]```   with ```-- master spark://master:7077```


### Other useful Commands
```jps``` - to verify the processes
```hdfs dfsadmin -report``` - to verify HDFS
```yarn node -list``` - to verify YARN



