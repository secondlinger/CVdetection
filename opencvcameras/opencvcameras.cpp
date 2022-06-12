#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>


using namespace cv;
using namespace std;
using namespace cv::dnn;

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.3;
const float NMS_THRESHOLD = 0.2;
const float CONFIDENCE_THRESHOLD = 0.3;

const float FONT_SCALE = 1;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 2;



void draw_label(Mat& input_image, string label, int left, int top)
{

    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw (0, 0, 0) rectangle.
    rectangle(input_image, tlc, brc, (0, 0, 0), FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, Scalar(255, 255, 255), THICKNESS);
}


vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // propagate 
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


Mat post_process(Mat& input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    vector<int> indices;
    int idx, left, top, width, height;
    Rect box;



    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);


    for (int i = 0; i < indices.size(); i++)
    {
        idx = indices[i];
        box = boxes[idx];

        left = box.x;
        top = box.y;
        width = box.width;
        height = box.height;
        rectangle(input_image, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 3 * THICKNESS);

        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        draw_label(input_image, label, left, top);
    }
    return input_image;
}



int main()
{
    
    //===========================================================================

    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    //===========================================================================

    namedWindow("Output", WINDOW_NORMAL);

    //===========================================================================

    // Load model.
    Net net;
    net = readNet("yolov5n.onnx");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);

    //===========================================================================

    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CAP_PROP_FRAME_HEIGHT, 1080);

    //===========================================================================

    // Load image.
    Mat frame;
    vector<Mat> detections;

    //===========================================================================
    while (1) {
        //===========================================================================

        cap >> frame;
        //===========================================================================

        resize(frame, frame, Size(640*2, 640*1));
        //===========================================================================

        detections = pre_process(frame, net);
        frame = post_process(frame, detections, class_list);
        //===========================================================================


        imshow("Output", frame);
        waitKey(1);
    }
    return 0;
}