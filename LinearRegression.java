import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

public class LinearRegression {

    // Mapper Class
    public static class LRM_Mapper extends Mapper<LongWritable, Text, Text, FloatWritable> {

        private final Text keyOut = new Text("key");
        private final FloatWritable valueOut = new FloatWritable();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (!line.isEmpty()) {
                float dataPoint = Float.parseFloat(line);
                valueOut.set(dataPoint);
                context.write(keyOut, valueOut);
            }
        }
    }

    // Reducer Class
    public static class LRM_Reducer extends Reducer<Text, FloatWritable, Text, FloatWritable> {

        private final FloatWritable result = new FloatWritable();

        public void reduce(Text key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {
            float sum = 0;
            float sumOfSquares = 0;
            int count = 0;

            for (FloatWritable val : values) {
                float dataPoint = val.get();
                sum += dataPoint;
                sumOfSquares += dataPoint * dataPoint;
                count++;
            }

            float mean = sum / count;
            float mse = sumOfSquares / count - mean * mean;
            result.set(mse);
            context.write(new Text("MSE"), result);
        }
    }

    // Main Method
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "linear regression");
        job.setJarByClass(LinearRegression.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapperClass(LRM_Mapper.class);
        job.setReducerClass(LRM_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FloatWritable.class);
        TextInputFormat.addInputPath(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

