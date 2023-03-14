package com.example.catsanddogspredictor;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    public int PICK_IMAGE_REQUEST = 1;
    public ImageView imageView;
    public TextView textView;
    Interpreter interpreter;
    public int result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.avatar);
        textView = findViewById(R.id.textView);

        try {
            interpreter = new Interpreter(loadModelFile());
            Toast.makeText(this, "Success", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Toast.makeText(this, e.toString(), Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }

       imageView.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {
               Intent intent = new Intent(Intent.ACTION_PICK);
               intent.setType("image/*");
               startActivityForResult(intent, PICK_IMAGE_REQUEST);
           }
       });
    }

    public MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                imageView.setImageBitmap(bitmap);

                // Pass the image to the model
                float[][] result = predictImage(bitmap);

                // Display the result
                if (result[0][0] > result[0][1]) {
                    textView.setText("Cat");
                } else {
                    textView.setText("Dog");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private float[][] predictImage(Bitmap bitmap) {
        // Resize and normalize the image
        int INPUT_IMAGE_SIZE = 128;
        Bitmap resizedBitmap = resize(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        ByteBuffer inputBuffer = normalizeImage(resizedBitmap);

        // Run the model inference
        float[][] output = new float[1][2]; // assuming binary classification
        interpreter.run(inputBuffer, output);
//        interpreter.run(inputBuffer, result);
//        Toast.makeText(this, output.toString(), Toast.LENGTH_SHORT).show();

        return output;
    }

    private Bitmap resize(Bitmap bitmap, int width, int height) {
        return Bitmap.createScaledBitmap(bitmap, width, height, true);
    }


    private ByteBuffer normalizeImage(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int channel = 3; // assuming RGB image

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * width * height * channel);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];

            // Extract the RGB components of the pixel
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            // Normalize the pixel values and add them to the input buffer
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        inputBuffer.rewind();

        return inputBuffer;
    }

}