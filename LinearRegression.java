import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class LinearRegression {

    // Mapper Class
    public static class LRM_Mapper extends Mapper<LongWritable, Text, Text, FloatWritable> {
        private final static FloatWritable one = new FloatWritable(1);
        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            word.set("key");
            FloatWritable dataPoint = new FloatWritable(Float.parseFloat(line));
            context.write(word, dataPoint);
        }
    }

    // Reducer Class
    public static class LRM_Reducer extends Reducer<Text, FloatWritable, Text, FloatWritable> {
        private FloatWritable result = new FloatWritable();

        public void reduce(Text key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {
            float sum = 0;
            float count = 0;
            for (FloatWritable val : values) {
                sum += val.get();
                count += 1;
            }
            float mean = sum / count;

            // Calculate the differences from the mean and square them
            float sumOfSquares = 0;
            for (FloatWritable val : values) {
                sumOfSquares += Math.pow(val.get() - mean, 2);
            }

            // Calculate MSE and Average MSE
            float mse = sumOfSquares / count;
            float avgMse = mse / count;

            result.set(mse);
            context.write(new Text("MSE"), result);
            result.set(avgMse);
            context.write(new Text("Average MSE"), result);
        }
    }

    // Main Method
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "linear regression");
        job.setJarByClass(LinearRegression.class);
        job.setMapperClass(LRM_Mapper.class);
        job.setCombinerClass(LRM_Reducer.class);
        job.setReducerClass(LRM_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FloatWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
