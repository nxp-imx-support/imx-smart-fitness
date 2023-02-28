/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * i.MX Smart Fitness application using GStreamer + NNStreamer
 *
 * Targets: i.MX8M Plus & i.MX93
 *
 * Smart Fitness demo shows the i.MX's Machine Learning capabilities
 * by using an NPU to accelerate two Deep Learning vision-based models.
 * Together, these models detect a person present in the scene and predict
 * 33 3D-keypoints to generate a complete body landmark. From this
 * landmark, a K-NN pose classifier is built to differenciate between two
 * different body poses: 'Squat-Down' and 'Squat-Up'. A counter shows the
 * number of repetitions the pearson has done for the 'Squat' fitness exercise.
 *
 * The solution is limited to classify and count only the 'squats' fitness
 * exercise and the repetition counting is set to 12 repetitions in an infinite
 * loop.
 *
 */

#include <cairo/cairo.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/gstbin.h>
#include <gst/gstbuffer.h>
#include <gst/gstelement.h>
#include <gst/gstpad.h>
#include <gst/gstpipeline.h>
#include <gst/video/video-info.h>
#include <nnstreamer/nnstreamer_util.h>

#include <csignal>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// cargs for argument parsing
#include "cargs/cargs.h"

// Classifier for squat pose
#include "classifier/pose_classification.h"
#include "classifier/repetition_counter.h"

// Mediapipe interpreters
#include "mediapipe/pose_detection_interpreter.h"
#include "mediapipe/pose_landmark_interpreter.h"
#include "utils/ema_filter.h"

#define WIDTH 640
#define HEIGHT 480

#define FONT_SIZE_LABEL_SCORE 10
#define FONT_SIZE_RUNTIME 35
#define INIT_POSITION_RUNTIME_STR 10

/**
 * Configuration for args
 */
static struct cag_option options[] = {

    {.identifier = 'd',
     .access_letters = "d",
     .access_name = "device",
     .value_name = "/dev/video<x>",
     .description = "Video source (camera): /dev/video<X> where X can change "
                    "depending on the hardware setup and target."},

    {.identifier = 't',
     .access_letters = "t",
     .access_name = "target",
     .value_name = "TARGET",
     .description = "i.MX target: i.MX8MP, i.MX93"},

    {.identifier = 'p',
     .access_letters = "p",
     .access_name = "pose-detection-model",
     .value_name = "./path/to/model.tflite",
     .description = "Path to pose detection TFlite model"},

    {.identifier = 'l',
     .access_letters = "l",
     .access_name = "pose-landmark-model",
     .value_name = "./path/to/model.tflite",
     .description = "Path to pose landmark TFlite model"},

    {.identifier = 'e',
     .access_letters = "e",
     .access_name = "pose-embeddings",
     .value_name = "./path/to/pose/embeddings.csv",
     .description = "Path to classification embeddings"},

    {.identifier = 'a',
     .access_letters = "a",
     .access_name = "anchors",
     .value_name = "./path/to/anchors.txt",
     .description = "Path to anchors file"},

    {.identifier = 'h',
     .access_letters = "h",
     .access_name = "help",
     .value_name = NULL,
     .description = "Shows the command help"}};

/**
 * Project configuration structure to store parsed information.
 */
struct configuration {
  bool camera_exists;
  bool target_exists;
  bool pose_detection_exists;
  bool pose_landmark_exists;
  bool pose_embeddings_exists;
  bool anchors_exists;
};

/**
 * Define structure to handle cairo overlay caps
 */
typedef struct {
  gboolean valid;
  GstVideoInfo vinfo;
} CairoOverlayState;

/**
 * Define the data structure to handle application
 */
typedef struct _appdata {
  // Main pipeline elements
  GstElement *pipeline;
  GstElement *tensor_sink_detection;
  GstElement *tensor_filter_pose;
  GstElement *appsink;
  GstElement *overlay;
  GstElement *wayland_sink;

  // Secondary pipeline elements
  GstElement *secondary_pipeline;
  GstElement *appsrc;
  GstElement *videocrop;
  GstElement *tensor_sink_landmark;
  GstElement *tensor_filter_landmark;
  guint source_id;

  GstBus *bus;
  GMainLoop *main_loop;
  GMutex g_mutex;

  CairoOverlayState overlay_state;

  BoundingBox pose;
  Landmark landmark;
  bool pose_detected;   // Flag for pose being detected
  int pad_img_shape[2]; // 0 width, 1 height

  // Pose bounding box variables
  guint top;
  guint left;
  guint bottom;
  guint right;
  guint roi_width;
  guint roi_height;
  guint roi_width_bbox;
  guint roi_height_bbox;

  // Define Interpreters
  PoseDetectionInterpreter *pose_detection_interpreter;
  PoseLandmarkInterpreter *pose_landmark_interpreter;
  guint inference_time_pose;
  guint inference_time_landmark;

  EMAFilter *filter_classification;
  Filter *filter_bbox;

  ClassificationResult result;
  RepetitionCounter *counter;

  PoseClassifier *classifier;

} AppData;

/**
 * Function handler for sigint signal to stop pipeline
 */
void sigint_handler(int signum);

/**
 * Function called when an error message is posted on the bus
 */
static void bus_message_callback(GstBus *bus, GstMessage *message,
                                 AppData *data);

/**
 * Get the caps information that need cairooverlay
 */
static void configure_overlay_callback(GstElement *overlay, GstCaps *caps,
                                       AppData *data);

/**
 * Function to handle tensorsink callback for pose detection
 */
static void new_pose_detection(GstElement *sink, GstBuffer *gstbuffer,
                               AppData *data);

/**
 * Function to handle appsink callback for pose landmarks
 */
static GstFlowReturn appsink_new_sample(GstElement *appsink, AppData *data);

/**
 * Function to handle appsrc callback for pose landmarks
 */
static void start_feed(GstElement *appsrc, guint size, AppData *data);

/**
 * Function to handle appsrc callback for pose landmarks
 */
static void stop_feed(GstElement *appsrc, AppData *data);

/**
 * Function to handle appsrc callback for pose landmarks
 */
static gboolean push_data(AppData *data);

/**
 * Function to handle tensor_sink callback for pose landmarks
 */
static void new_pose_landmarks(GstElement *sink, GstBuffer *gstbuffer,
                               AppData *data);

/**
 * Function to handle cairooverlay callback to draw results
 */
static void draw_overlay(GstElement *overlay, cairo_t *cr, guint64 timestamp,
                         guint64 duration, AppData *data);

/**
 * Funtion to compute preprocess of input frame
 */
void preprocess_input_frame(const int &video_width, const int &video_height,
                            int &scaled_width, int &scaled_height,
                            AppData *data);

/**
 * Funtion to draw bbox and landmarks
 */
void draw_detections(AppData *data, cairo_t *cr);

/**
 * Global data structure for pipeline
 */
static AppData data = {};

/**
 * Main function that runs the demo
 */
int main(int argc, char *argv[]) {

  // Variables for arguments
  char identifier;
  const gchar *camera = nullptr;
  const gchar *target = nullptr;
  const gchar *pose_detection_model = nullptr;
  const gchar *pose_landmark_model = nullptr;
  const char *pose_embeddings = nullptr;
  const gchar *anchors = nullptr;
  cag_option_context context;
  struct configuration config = {false, false, false, false, false, false};

  cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
  while (cag_option_fetch(&context)) {
    identifier = cag_option_get(&context);
    switch (identifier) {
    case 'd':
      config.camera_exists = true;
      camera = cag_option_get_value(&context);
      break;
    case 't':
      config.target_exists = true;
      target = cag_option_get_value(&context);
      break;
    case 'p':
      config.pose_detection_exists = true;
      pose_detection_model = cag_option_get_value(&context);
      break;
    case 'l':
      config.pose_landmark_exists = true;
      pose_landmark_model = cag_option_get_value(&context);
      break;
    case 'e':
      config.pose_embeddings_exists = true;
      pose_embeddings = cag_option_get_value(&context);
      break;
    case 'a':
      config.anchors_exists = true;
      anchors = cag_option_get_value(&context);
      break;
    case 'h':
      printf("Usage: imx-smart-fitness [OPTION]...\n");
      printf("i.MX Smart Fitness Application.\n\n");
      cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
      return EXIT_SUCCESS;
    }
  }

  if (!config.camera_exists) {
    std::cerr << "Please provide a valid video source device.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  if (!config.target_exists) {
    std::cerr << "Please provide a valid target.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  if (!config.pose_detection_exists) {
    std::cerr << "Please provide the path to the TFLite pose detection model.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  if (!config.pose_landmark_exists) {
    std::cerr << "Please provide the path to the TFLite pose landmark model.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  if (!config.pose_embeddings_exists) {
    std::cerr << "Please provide the path to the pose embeddings file.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  if (!config.anchors_exists) {
    std::cerr << "Please provide the path to the anchors file.\n"
                 "Run \'./imx-smart-fitness --help\' for more information.\n";
    return EXIT_FAILURE;
  }

  // Define delegate and converter for selected target
  const char *delegate = nullptr;
  const char *nxp_converter = nullptr;
  if (strcmp(target, "i.MX8MP") == 0) {
    nxp_converter = "imxvideoconvert_g2d";
    delegate = "libvx_delegate.so";
  } else if (strcmp(target, "i.MX93") == 0) {
    nxp_converter = "imxvideoconvert_pxp";
    delegate = "libethosu_delegate.so";
  } else {
    g_printerr("Target not supported!\n");
  }

  // Register signal SIGINT and signal handler
  signal(SIGINT, sigint_handler);

  // Initialize GStreamer
  gst_init(&argc, &argv);

  /* Initialize elements to nullptr */

  // Main pipeline elements
  data.pipeline = nullptr;
  data.tensor_sink_detection = nullptr;
  data.tensor_filter_pose = nullptr;
  data.appsink = nullptr;
  data.overlay = nullptr;
  data.wayland_sink = nullptr;

  // Secondary pipeline elements
  data.secondary_pipeline = nullptr;
  data.appsrc = nullptr;
  data.videocrop = nullptr;
  data.tensor_sink_landmark = nullptr;
  data.tensor_filter_landmark = nullptr;
  data.source_id = 0;

  // Shared elements
  data.bus = nullptr;
  data.main_loop = nullptr;
  g_mutex_init(&data.g_mutex);

  // MediaPipe interpreters
  data.pose_detection_interpreter = new PoseDetectionInterpreter(anchors);
  data.pose_landmark_interpreter = new PoseLandmarkInterpreter();
  data.inference_time_pose = 0;
  data.inference_time_landmark = 0;
  data.pose_detected = false;

  // Pose bounding box variables
  data.top = 0;
  data.left = 0;
  data.bottom = 0;
  data.right = 0;
  data.roi_width = 0;
  data.roi_height = 0;
  data.roi_width_bbox = 0;
  data.roi_height_bbox = 0;

  data.filter_classification = new EMAFilter();
  data.filter_bbox = new Filter();
  data.counter = new RepetitionCounter("squats_down");

  data.classifier = new PoseClassifier(pose_embeddings);

  // Video input size and scaled size
  memset(data.pad_img_shape, 0, 2 * sizeof(int));
  int video_width = WIDTH;
  int video_height = HEIGHT;
  int scaled_width = 0;
  int scaled_height = 0;
  preprocess_input_frame(video_width, video_height, scaled_width, scaled_height,
                         &data);

  // Create GLib main loop and set it to run
  data.main_loop = g_main_loop_new(NULL, FALSE);
  if (!data.main_loop) {
    g_printerr("Failed to create main loop!\n");
    return EXIT_FAILURE;
  }

  // Create pipeline
  gchar *pipeline_cmd = g_strdup_printf(
      "v4l2src device=%s ! "
      "video/x-raw,width=%d,height=%d,framerate=30/1,format=YUY2 ! "
      "tee name=t "
      // Pose detection
      "t. ! queue max-size-buffers=1 leaky=1 ! "
      "videobox autocrop=false bottom=-160 ! "
      "%s ! video/x-raw,width=224,height=224 ! "
      "videoconvert ! video/x-raw,format=RGB ! "
      "tensor_converter ! "
      "tensor_transform mode=arithmetic "
      "option=typecast:float32,div:255.0,add:-0.5,mul:2.0 ! "
      "tensor_filter framework=tensorflow-lite "
      "model=%s "
      "accelerator=true:npu "
      "custom=Delegate:External,ExtDelegateLib:%s "
      "name=tensor_filter_pose ! "
      "tensor_sink name=tensor_sink "
      // Pose landmarks
      "t. ! queue max-size-buffers=1 leaky=2 ! "
      "appsink name=appsink max-buffers=1 "
      // Draw results on screen
      "t. ! queue max-size-buffers=1 leaky=1 ! %s ! "
      "cairooverlay name=overlay ! "
      "fpsdisplaysink name=fps_sink text-overlay=false video-sink=waylandsink "
      "sync=false",
      camera, video_width, video_height, nxp_converter, pose_detection_model,
      delegate, nxp_converter);

  // Create secondary pipeline for pose landmarks
  gchar *secondary_pipeline_cmd = g_strdup_printf(
      "appsrc name=appsrc_video "
      "max-buffers=1 leaky_type=2 format=3 "
      "caps=video/x-raw,width=%d,height=%d,framerate=30/1,format=YUY2 ! "
      "video/x-raw,width=%d,height=%d,framerate=30/1 ! "
      "videocrop name=video_crop ! "
      "%s ! video/x-raw,width=256,height=256 ! "
      "videoconvert ! video/x-raw,format=RGB ! "
      "tensor_converter ! "
      "tensor_transform mode=arithmetic "
      "option=typecast:float32,div:255.0 ! "
      "tensor_filter framework=tensorflow-lite "
      "model=%s "
      "accelerator=true:npu "
      "custom=Delegate:External,ExtDelegateLib:%s "
      "name=tensor_filter_landmark ! "
      "tensor_sink name=second_tensor_sink",
      video_width, video_height, video_width, video_height, nxp_converter,
      pose_landmark_model, delegate);

  // Parse main pipeline
  data.pipeline = gst_parse_launch(pipeline_cmd, NULL);
  g_free(pipeline_cmd);

  // Parse secondary pipeline
  data.secondary_pipeline = gst_parse_launch(secondary_pipeline_cmd, NULL);
  g_free(secondary_pipeline_cmd);

  /* SET UP PRIMARY PIPELINE ELEMENTS */

  // Add callback to tensor_sink for pose detection
  data.tensor_sink_detection =
      gst_bin_get_by_name(GST_BIN(data.pipeline), "tensor_sink");
  g_object_set(GST_OBJECT(data.tensor_sink_detection), "emit-signal",
               (gboolean)TRUE, NULL);
  g_signal_connect(GST_OBJECT(data.tensor_sink_detection), "new-data",
                   G_CALLBACK(new_pose_detection), &data);
  gst_object_unref(GST_OBJECT(data.tensor_sink_detection));

  // Add callback to appsink for pose landmark
  data.appsink = gst_bin_get_by_name(GST_BIN(data.pipeline), "appsink");
  g_object_set(G_OBJECT(data.appsink), "emit-signals", (gboolean)TRUE, "sync",
               (gboolean)FALSE, "drop", (gboolean)TRUE, NULL);
  g_signal_connect(GST_OBJECT(data.appsink), "new-sample",
                   G_CALLBACK(appsink_new_sample), &data);
  gst_object_unref(GST_OBJECT(data.appsink));

  // Add callback to cairooverlay for drawing results to screen
  data.overlay = gst_bin_get_by_name(GST_BIN(data.pipeline), "overlay");
  g_signal_connect(GST_OBJECT(data.overlay), "draw", G_CALLBACK(draw_overlay),
                   &data);
  g_signal_connect(GST_OBJECT(data.overlay), "caps-changed",
                   G_CALLBACK(configure_overlay_callback), &data);
  gst_object_unref(GST_OBJECT(data.overlay));

  // Get latency property from pose_detection
  data.tensor_filter_pose =
      gst_bin_get_by_name(GST_BIN(data.pipeline), "tensor_filter_pose");
  g_object_set(data.tensor_filter_pose, "latency", 1, NULL);
  gst_object_unref(GST_OBJECT(data.tensor_filter_pose));

  // Get FPS from waylandsink
  data.wayland_sink = gst_bin_get_by_name(GST_BIN(data.pipeline), "fps_sink");
  gst_object_unref(GST_OBJECT(data.wayland_sink));

  // Add bus for message handling of pipeline
  data.bus = gst_pipeline_get_bus(GST_PIPELINE(data.pipeline));
  gst_bus_add_signal_watch(data.bus);
  g_signal_connect(data.bus, "message", G_CALLBACK(bus_message_callback),
                   &data);
  gst_object_unref(GST_OBJECT(data.bus));

  /* SET UP SECONDARY PIPELINE ELEMENTS */

  // Add callback to tensor_sink for pose landmark
  data.tensor_sink_landmark = gst_bin_get_by_name(
      GST_BIN(data.secondary_pipeline), "second_tensor_sink");
  g_object_set(GST_OBJECT(data.tensor_sink_landmark), "emit-signal",
               (gboolean)TRUE, NULL);
  g_signal_connect(GST_OBJECT(data.tensor_sink_landmark), "new-data",
                   G_CALLBACK(new_pose_landmarks), &data);
  gst_object_unref(GST_OBJECT(data.tensor_sink_landmark));

  // Add callback to appsrc for pose landmark
  data.appsrc =
      gst_bin_get_by_name(GST_BIN(data.secondary_pipeline), "appsrc_video");
  g_object_set(G_OBJECT(data.appsrc), "is-live", (gboolean)TRUE, "stream-type",
               0, NULL);
  g_signal_connect(GST_OBJECT(data.appsrc), "need-data", G_CALLBACK(start_feed),
                   &data);
  g_signal_connect(GST_OBJECT(data.appsrc), "enough-data",
                   G_CALLBACK(stop_feed), &data);
  gst_object_unref(GST_OBJECT(data.appsrc));

  // Videocrop for pose-landmarks
  data.videocrop =
      gst_bin_get_by_name(GST_BIN(data.secondary_pipeline), "video_crop");
  gst_object_unref(GST_OBJECT(data.videocrop));

  // Get latency property from pose_landmark
  data.tensor_filter_landmark = gst_bin_get_by_name(
      GST_BIN(data.secondary_pipeline), "tensor_filter_landmark");
  g_object_set(data.tensor_filter_landmark, "latency", 1, NULL);
  gst_object_unref(GST_OBJECT(data.tensor_filter_landmark));

  // Add bus for message handling of secondary pipeline
  data.bus = gst_pipeline_get_bus(GST_PIPELINE(data.secondary_pipeline));
  gst_bus_add_signal_watch(data.bus);
  g_signal_connect(data.bus, "message", G_CALLBACK(bus_message_callback),
                   &data);
  gst_object_unref(GST_OBJECT(data.bus));

  /* SET TO PLAYING STATE */

  // Set pipeline to playing state
  g_print("Setting pipeline to PLAYING...\n");
  gst_element_set_state(data.pipeline, GST_STATE_PLAYING);

  // Set secondary pipeline to playing state
  g_print("Setting secondary pipeline to PLAYING...\n");
  gst_element_set_state(data.secondary_pipeline, GST_STATE_PLAYING);

  // Set pipeline to run main loop
  g_main_loop_run(data.main_loop);

  // Quit when received error or EOS message (primary pipeline)
  g_print("Setting pipeline to PAUSED...\n");
  gst_element_set_state(data.pipeline, GST_STATE_PAUSED);

  g_print("Setting pipeline to READY...\n");
  gst_element_set_state(data.pipeline, GST_STATE_READY);

  g_print("Setting pipeline to NULL...\n");
  gst_element_set_state(data.pipeline, GST_STATE_NULL);

  // Quit when received error or EOS message (secondary pipeline)
  g_print("Setting secondary pipeline to PAUSED...\n");
  gst_element_set_state(data.secondary_pipeline, GST_STATE_PAUSED);

  g_print("Setting secondary pipeline to READY...\n");
  gst_element_set_state(data.secondary_pipeline, GST_STATE_READY);

  g_print("Setting secondary pipeline to NULL...\n");
  gst_element_set_state(data.secondary_pipeline, GST_STATE_NULL);

  gst_object_unref(data.pipeline);
  data.pipeline = nullptr;
  gst_object_unref(data.secondary_pipeline);
  data.secondary_pipeline = nullptr;

  g_main_loop_unref(data.main_loop);
  data.main_loop = nullptr;

  gst_bus_remove_signal_watch(data.bus);
  data.bus = nullptr;

  g_mutex_clear(&data.g_mutex);

  delete data.pose_detection_interpreter;
  delete data.pose_landmark_interpreter;
  delete data.filter_classification;
  delete data.filter_bbox;
  delete data.counter;
  delete data.classifier;

  data.pose_detection_interpreter = nullptr;
  data.pose_landmark_interpreter = nullptr;
  data.filter_classification = nullptr;
  data.filter_bbox = nullptr;
  data.counter = nullptr;
  data.classifier = nullptr;

  return EXIT_SUCCESS;
}

/**
 * Function called when an error message is posted on the bus
 */
static void bus_message_callback(GstBus *bus, GstMessage *message,
                                 AppData *data) {
  UNUSED(bus);
  switch (GST_MESSAGE_TYPE(message)) {
  case GST_MESSAGE_EOS: {
    // End of stream
    g_print("\nGot EOS from element \"%s\".\n", GST_MESSAGE_SRC_NAME(message));
    g_main_loop_quit(data->main_loop);
    break;
  }

  case GST_MESSAGE_ERROR: {
    GError *err;
    gchar *debug_info;

    gst_message_parse_error(message, &err, &debug_info);
    g_printerr("Error message received from element [%s]: %s\n",
               GST_MESSAGE_SRC_NAME(message), err->message);
    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error(&err);
    g_free(debug_info);

    g_main_loop_quit(data->main_loop);
    break;
  }

  case GST_MESSAGE_WARNING: {
    GError *err;
    gchar *debug_info;

    gst_message_parse_warning(message, &err, &debug_info);
    g_printerr("Warning message received from element [%s]: %s\n",
               GST_MESSAGE_SRC_NAME(message), err->message);
    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error(&err);
    g_free(debug_info);
    break;
  }

  case GST_MESSAGE_QOS: {
    GstFormat format;
    guint64 processed;
    guint64 dropped;

    g_print("Got %s message from element %s\n", GST_MESSAGE_TYPE_NAME(message),
            GST_MESSAGE_SRC_NAME(message));

    gst_message_parse_qos_stats(message, &format, &processed, &dropped);
    g_print("Format: [%d], processed: [%" G_GUINT64_FORMAT
            "], dropped: [%" G_GUINT64_FORMAT "]\n",
            format, processed, dropped);
    break;
  }

  default:
    break;
  }
}

/**
 * Get the caps information that need cairooverlay
 */
static void configure_overlay_callback(GstElement *overlay, GstCaps *caps,
                                       AppData *data) {
  UNUSED(overlay);
  CairoOverlayState *state = &(data->overlay_state);
  state->valid = gst_video_info_from_caps(&state->vinfo, caps);
}

/**
 * Function to handle tensorsink callback for pose detection
 */
static void new_pose_detection(GstElement *sink, GstBuffer *gstbuffer,
                               AppData *data) {
  UNUSED(sink);
  GstMemory *mem = NULL;
  GstMapInfo info;

  float *raw_scores = nullptr;
  float *bbox_detection = nullptr;

  for (size_t i{0}; i < gst_buffer_n_memory(gstbuffer); i++) {
    g_mutex_lock(&data->g_mutex);
    mem = gst_buffer_peek_memory(gstbuffer, i);
    g_mutex_unlock(&data->g_mutex);
    if (mem != NULL) {
      if (gst_memory_map(mem, &info, GST_MAP_READ)) {
        size_t length = info.size / 4;
        if (length == 2254) {
          raw_scores = (float *)(info.data);
        }
        if (length == 27048) {
          bbox_detection = (float *)(info.data);
        }
        gst_memory_unmap(mem, &info);
      }
    } else {
      g_printerr("Failed peek memory\n");
    }
  }

  data->pose_detection_interpreter->decode_predictions(bbox_detection,
                                                       raw_scores);
}

/**
 * Function to handle appsink callback for pose landmarks
 */
static GstFlowReturn appsink_new_sample(GstElement *appsink, AppData *data) {
  GstSample *sample;
  GstBuffer *buffer;
  GstFlowReturn ret;

  // Get sample from appsink
  sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
  buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    g_printerr("Got NULL buffer from sample! Exiting...\n");
    return GST_FLOW_EOS;
  }

  if (data->left > 0 && data->top > 0 && data->pose("xmax") < WIDTH &&
      data->pose("ymax") < HEIGHT) {
    if (data->roi_width_bbox > 0 && data->left + data->roi_width_bbox < WIDTH &&
        data->roi_height_bbox > 0 &&
        data->top + data->roi_height_bbox < HEIGHT) {
      // Update size for cropping bbox for pose detection
      g_mutex_lock(&data->g_mutex);
      g_object_set(G_OBJECT(data->videocrop), "left", data->left, "right",
                   data->right, "top", data->top, "bottom", data->bottom, NULL);
      g_signal_emit_by_name(data->appsrc, "push-buffer", buffer, &ret);
      g_mutex_unlock(&data->g_mutex);
      data->pose_detected = true;
    }
  } else {
    data->pose_detected = false;
    // No landmark will be detected; add empty result
    ClassificationResult empty;
    data->result = data->filter_classification->filter(empty);
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

/**
 * Function to handle appsrc callback for pose landmarks
 */
static void start_feed(GstElement *appsrc, guint size, AppData *data) {
  UNUSED(appsrc);
  UNUSED(size);
  if (data->source_id == 0) {
    g_print("Start feeding frames to secondary pipeline...\n");
    data->source_id = g_idle_add((GSourceFunc)push_data, data);
  }
}

/**
 * Function to handle appsrc callback for pose landmarks
 */
static void stop_feed(GstElement *source, AppData *data) {
  UNUSED(source);
  if (data->source_id != 0) {
    g_print("Stop feeding...\n");
    g_source_remove(data->source_id);
    data->source_id = 0;
  }
}

/**
 * Function to handle appsrc callback for pose landmarks
 */
static gboolean push_data(AppData *data) {
  UNUSED(data);
  return TRUE;
}

/**
 * Function to handle tensor_sink callback for pose landmarks
 */
static void new_pose_landmarks(GstElement *sink, GstBuffer *gstbuffer,
                               AppData *data) {
  UNUSED(sink);
  GstMemory *mem = NULL;
  GstMapInfo info;

  float *score = nullptr;
  float *raw_landmark = nullptr;

  for (size_t i{0}; i < gst_buffer_n_memory(gstbuffer); i++) {
    g_mutex_lock(&data->g_mutex);
    mem = gst_buffer_peek_memory(gstbuffer, i);
    g_mutex_unlock(&data->g_mutex);
    if (mem != NULL) {
      if (gst_memory_map(mem, &info, GST_MAP_READ)) {
        size_t length = info.size / 4;
        if (length == 195) {
          raw_landmark = (float *)(info.data);
        }
        if (length == 1) {
          score = (float *)(info.data);
        }
        gst_memory_unmap(mem, &info);
      }
    } else {
      g_printerr("Failed peek memory\n");
    }
  }

  data->pose_landmark_interpreter->decode_predictions(raw_landmark, *score);
  data->landmark = data->pose_landmark_interpreter->get_pose_landmark();
  data->landmark = data->filter_bbox->filter(data->landmark);

  ClassificationResult classification_result =
      data->classifier->classify_pose(data->landmark);
  data->result = data->filter_classification->filter(classification_result);
}

/**
 * Function to handle cairooverlay callback to draw results
 */
static void draw_overlay(GstElement *overlay, cairo_t *cr, guint64 timestamp,
                         guint64 duration, AppData *data) {
  UNUSED(overlay);
  UNUSED(timestamp);
  UNUSED(duration);
  CairoOverlayState *state = &(data->overlay_state);

  if (state->valid == TRUE) {
    // Set Cairo config
    cairo_set_line_width(cr, 3);
    cairo_select_font_face(cr, "Courier", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, FONT_SIZE_LABEL_SCORE);

    // Draw runtime string
    char runtime_str[256];

    gchar *fps_msg = nullptr;
    g_object_get(G_OBJECT(data->wayland_sink), "last-message", &fps_msg, NULL);

    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_move_to(cr, 10, INIT_POSITION_RUNTIME_STR);
    snprintf(runtime_str, sizeof(runtime_str), "FRAME INFO: %s", fps_msg);
    cairo_show_text(cr, runtime_str);

    // Get pose detection inference time in us
    g_object_get(G_OBJECT(data->tensor_filter_pose), "latency",
                 &data->inference_time_pose, NULL);

    cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
    cairo_move_to(cr, 10, INIT_POSITION_RUNTIME_STR + 10);
    snprintf(runtime_str, sizeof(runtime_str),
             "Pose detection avg. inference time: %.2f ms",
             data->inference_time_pose / 1000.0);
    cairo_show_text(cr, runtime_str);

    // Get pose landmark inference time in us
    g_object_get(G_OBJECT(data->tensor_filter_landmark), "latency",
                 &data->inference_time_landmark, NULL);

    cairo_move_to(cr, 10, INIT_POSITION_RUNTIME_STR + 20);
    snprintf(runtime_str, sizeof(runtime_str),
             "Pose landmark avg. inference time: %.2f ms",
             (data->pose_detected ? (data->inference_time_landmark / 1000.0)
                                  : 0.00));
    cairo_show_text(cr, runtime_str);

    cairo_set_font_size(cr, FONT_SIZE_RUNTIME + 2);
    cairo_move_to(cr, 10, INIT_POSITION_RUNTIME_STR + HEIGHT - 55);
    cairo_set_source_rgb(
        cr, 1.0 - data->result.get_class_confidence("squats_up") / 10.0,
        data->result.get_class_confidence("squats_up") / 10.0, 0.0);
    snprintf(runtime_str, sizeof(runtime_str), "Squat-Up");
    cairo_show_text(cr, runtime_str);

    cairo_move_to(cr, WIDTH - 230, INIT_POSITION_RUNTIME_STR + HEIGHT - 55);
    cairo_set_source_rgb(
        cr, 1.0 - data->result.get_class_confidence("squats_down") / 10.0,
        data->result.get_class_confidence("squats_down") / 10.0, 0.0);
    snprintf(runtime_str, sizeof(runtime_str), "Squat-Down");
    cairo_show_text(cr, runtime_str);

    // Draw count
    cairo_set_font_size(cr, FONT_SIZE_RUNTIME + 20);
    cairo_move_to(cr, WIDTH - 100, INIT_POSITION_RUNTIME_STR + 40);
    cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
    snprintf(runtime_str, sizeof(runtime_str), "x%d",
             data->counter->count(data->result));
    cairo_show_text(cr, runtime_str);

    // Draw graph
    cairo_set_line_width(cr, 15);
    cairo_set_source_rgb(
        cr, 1.0 - data->result.get_class_confidence("squats_down") / 10.0,
        data->result.get_class_confidence("squats_down") / 10.0, 0.0);
    cairo_move_to(cr, 0, HEIGHT - 15);
    cairo_line_to(cr, data->result.get_class_confidence("squats_down") * 64,
                  HEIGHT - 15);

    cairo_stroke(cr);
    draw_detections(data, cr);
  }
}

/**
 * Funtion to compute preprocess of input frame
 */
void preprocess_input_frame(const int &video_width, const int &video_height,
                            int &scaled_width, int &scaled_height,
                            AppData *data) {
  int input_height = 224;
  int input_width = 224;

  // Keep original aspect ratio to scale frame
  float scale_w = static_cast<float>(video_width) / input_width;
  float scale_h = static_cast<float>(video_height) / input_height;
  float ratio = static_cast<float>(video_width) / video_height;
  float input_ratio = static_cast<float>(input_width) / input_height;

  if (scale_w > scale_h) {
    scaled_width = input_width;
    scaled_height = static_cast<int>(input_width / ratio);
    data->pad_img_shape[0] = video_width;
    data->pad_img_shape[1] = int(video_width / input_ratio);
  } else {
    scaled_height = input_height;
    scaled_width = static_cast<int>(input_height * ratio);
    data->pad_img_shape[1] = video_height;
    data->pad_img_shape[0] = int(video_height * input_ratio);
  }
}

/**
 * Funtion to draw bbox and landmarks
 */
void draw_detections(AppData *data, cairo_t *cr) {

  // Recover box location after resizing
  Keypoint pad_bbox(data->pad_img_shape[0], data->pad_img_shape[1]);
  Keypoint center_bbox(WIDTH / 2, HEIGHT / 2);

  PoseDetection pose;
  int min_distance = WIDTH;
  int index_bbox{-1};

  // Get closest bbox to center of frame and only process for one person
  for (size_t i{0};
       i < data->pose_detection_interpreter->get_pose_detections().size();
       i++) {
    pose = data->pose_detection_interpreter->get_pose_detections().at(i);

    Keypoint mid_hip_center(pose.get_mid_hip_center());
    float distance = (mid_hip_center * pad_bbox) ^
                     center_bbox; // Get distance between points
    if (distance < min_distance) {
      min_distance = distance;
      index_bbox = i;
    }
  }

  if (index_bbox >= 0) {
    // Compute radius of body for bounding box
    pose =
        data->pose_detection_interpreter->get_pose_detections().at(index_bbox);
    float radius =
        (pose.get_full_body_size_rotation() ^ pose.get_mid_hip_center()) *
        pad_bbox["y"];

    // Create main pose bbox
    BoundingBox tmp((pose.get_mid_hip_center() * pad_bbox) - radius,
                    (pose.get_mid_hip_center() * pad_bbox) + radius);

    // Filter bounding box
    data->pose = data->filter_bbox->filter(tmp);

    data->top = data->pose("ymin");
    data->left = data->pose("xmin");
    data->bottom = HEIGHT - data->pose("ymax");
    data->right = WIDTH - data->pose("xmax");

    data->roi_width_bbox = data->right - data->left;
    data->roi_height_bbox = data->bottom - data->top;

    data->roi_width = data->pose("xmax") - data->left;
    data->roi_height = data->pose("ymax") - data->top;

    if (data->pose_detected) {
      // Set color to green to show that landmarks will be printed
      cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);
    } else {
      // Set color to red to show that landmarks won't be printed.
      cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
    }

    cairo_set_line_width(cr, 3);

    //** Draw Pose Box **
    cairo_move_to(cr, data->left, data->top);
    cairo_line_to(cr, data->pose("xmax"), data->top);
    cairo_line_to(cr, data->pose("xmax"), data->pose("ymax"));
    cairo_line_to(cr, data->left, data->pose("ymax"));
    cairo_close_path(cr);
    cairo_stroke(cr);

    //** Draw landmarks **
    if (data->pose_detected) {
      cairo_set_source_rgb(cr, 1.0, 0.64705882, 0.0); // Orange
      cairo_move_to(cr,
                    data->landmark["nose"]["x"] * data->roi_width + data->left,
                    data->landmark["nose"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_eye"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_ear"]["x"] * data->roi_width + data->left,
          data->landmark["left_ear"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_set_source_rgb(cr, 0.0, 1.0, 1.0); // Blue
      cairo_move_to(cr,
                    data->landmark["nose"]["x"] * data->roi_width + data->left,
                    data->landmark["nose"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_eye"]["x"] * data->roi_width + data->left,
          data->landmark["right_eye"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_ear"]["x"] * data->roi_width + data->left,
          data->landmark["right_ear"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_move_to(
          cr, data->landmark["right_thumb"]["x"] * data->roi_width + data->left,
          data->landmark["right_thumb"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["right_wrist"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_index"]["x"] * data->roi_width + data->left,
          data->landmark["right_index"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_pinky"]["x"] * data->roi_width + data->left,
          data->landmark["right_pinky"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["right_wrist"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["right_elbow"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr,
          data->landmark["right_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["right_shoulder"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_set_source_rgb(cr, 1.0, 0.64705882, 0.0); // Orange
      cairo_move_to(
          cr,
          data->landmark["left_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["left_shoulder"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["left_elbow"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["left_wrist"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_pinky"]["x"] * data->roi_width + data->left,
          data->landmark["left_pinky"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_index"]["x"] * data->roi_width + data->left,
          data->landmark["left_index"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["left_wrist"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_thumb"]["x"] * data->roi_width + data->left,
          data->landmark["left_thumb"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_move_to(
          cr,
          data->landmark["left_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["left_shoulder"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_hip"]["x"] * data->roi_width + data->left,
          data->landmark["left_hip"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_knee"]["x"] * data->roi_width + data->left,
          data->landmark["left_knee"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["left_ankle"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_heel"]["x"] * data->roi_width + data->left,
          data->landmark["left_heel"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_foot"]["x"] * data->roi_width + data->left,
          data->landmark["left_foot"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["left_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["left_ankle"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_set_source_rgb(cr, 0.0, 1.0, 1.0); // Blue
      cairo_move_to(
          cr,
          data->landmark["right_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["right_shoulder"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_hip"]["x"] * data->roi_width + data->left,
          data->landmark["right_hip"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_knee"]["x"] * data->roi_width + data->left,
          data->landmark["right_knee"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["right_ankle"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_heel"]["x"] * data->roi_width + data->left,
          data->landmark["right_heel"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_foot"]["x"] * data->roi_width + data->left,
          data->landmark["right_foot"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["right_ankle"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); // White
      cairo_move_to(
          cr, data->landmark["mouth_left"]["x"] * data->roi_width + data->left,
          data->landmark["mouth_left"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["mouth_right"]["x"] * data->roi_width + data->left,
          data->landmark["mouth_right"]["y"] * data->roi_height + data->top);
      cairo_move_to(
          cr,
          data->landmark["left_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["left_shoulder"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr,
          data->landmark["right_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["right_shoulder"]["y"] * data->roi_height + data->top);
      cairo_move_to(
          cr, data->landmark["left_hip"]["x"] * data->roi_width + data->left,
          data->landmark["left_hip"]["y"] * data->roi_height + data->top);
      cairo_line_to(
          cr, data->landmark["right_hip"]["x"] * data->roi_width + data->left,
          data->landmark["right_hip"]["y"] * data->roi_height + data->top);
      cairo_stroke(cr);

      // Draw points
      cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
      cairo_arc(
          cr, data->landmark["mouth_left"]["x"] * data->roi_width + data->left,
          data->landmark["mouth_left"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["mouth_right"]["x"] * data->roi_width + data->left,
          data->landmark["mouth_right"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["mouth_right"]["x"] * data->roi_width + data->left,
          data->landmark["mouth_right"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr,
          data->landmark["left_eye_inner"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye_inner"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr,
          data->landmark["left_eye_inner"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye_inner"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_eye"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_eye"]["x"] * data->roi_width + data->left,
                data->landmark["left_eye"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr,
          data->landmark["left_eye_outer"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye_outer"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr,
          data->landmark["left_eye_outer"]["x"] * data->roi_width + data->left,
          data->landmark["left_eye_outer"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_ear"]["x"] * data->roi_width + data->left,
          data->landmark["left_ear"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_ear"]["x"] * data->roi_width + data->left,
                data->landmark["left_ear"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(cr,
                    data->landmark["right_eye_inner"]["x"] * data->roi_width +
                        data->left,
                    data->landmark["right_eye_inner"]["y"] * data->roi_height +
                        data->top);
      cairo_arc(
          cr,
          data->landmark["right_eye_inner"]["x"] * data->roi_width + data->left,
          data->landmark["right_eye_inner"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_ear"]["x"] * data->roi_width + data->left,
          data->landmark["right_ear"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["right_ear"]["x"] * data->roi_width + data->left,
                data->landmark["right_ear"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(cr,
                    data->landmark["right_eye_outer"]["x"] * data->roi_width +
                        data->left,
                    data->landmark["right_eye_outer"]["y"] * data->roi_height +
                        data->top);
      cairo_arc(
          cr,
          data->landmark["right_eye_outer"]["x"] * data->roi_width + data->left,
          data->landmark["right_eye_outer"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_ear"]["x"] * data->roi_width + data->left,
          data->landmark["right_ear"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["right_ear"]["x"] * data->roi_width + data->left,
                data->landmark["right_ear"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr,
          data->landmark["left_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["left_shoulder"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr,
          data->landmark["left_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["left_shoulder"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr,
          data->landmark["right_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["right_shoulder"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr,
          data->landmark["right_shoulder"]["x"] * data->roi_width + data->left,
          data->landmark["right_shoulder"]["y"] * data->roi_height + data->top,
          1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_hip"]["x"] * data->roi_width + data->left,
          data->landmark["left_hip"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_hip"]["x"] * data->roi_width + data->left,
                data->landmark["left_hip"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_hip"]["x"] * data->roi_width + data->left,
          data->landmark["right_hip"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["right_hip"]["x"] * data->roi_width + data->left,
                data->landmark["right_hip"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_knee"]["x"] * data->roi_width + data->left,
          data->landmark["left_knee"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_knee"]["x"] * data->roi_width + data->left,
                data->landmark["left_knee"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_knee"]["x"] * data->roi_width + data->left,
          data->landmark["right_knee"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_knee"]["x"] * data->roi_width + data->left,
          data->landmark["right_knee"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["left_ankle"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["left_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["left_ankle"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["right_ankle"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_ankle"]["x"] * data->roi_width + data->left,
          data->landmark["right_ankle"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_heel"]["x"] * data->roi_width + data->left,
          data->landmark["left_heel"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_heel"]["x"] * data->roi_width + data->left,
                data->landmark["left_heel"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_heel"]["x"] * data->roi_width + data->left,
          data->landmark["right_heel"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_heel"]["x"] * data->roi_width + data->left,
          data->landmark["right_heel"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_foot"]["x"] * data->roi_width + data->left,
          data->landmark["left_foot"]["y"] * data->roi_height + data->top);
      cairo_arc(cr,
                data->landmark["left_foot"]["x"] * data->roi_width + data->left,
                data->landmark["left_foot"]["y"] * data->roi_height + data->top,
                1, 0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_foot"]["x"] * data->roi_width + data->left,
          data->landmark["right_foot"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_foot"]["x"] * data->roi_width + data->left,
          data->landmark["right_foot"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["left_elbow"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["left_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["left_elbow"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["right_elbow"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_elbow"]["x"] * data->roi_width + data->left,
          data->landmark["right_elbow"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["left_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["left_wrist"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["left_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["left_wrist"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_move_to(
          cr, data->landmark["right_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["right_wrist"]["y"] * data->roi_height + data->top);
      cairo_arc(
          cr, data->landmark["right_wrist"]["x"] * data->roi_width + data->left,
          data->landmark["right_wrist"]["y"] * data->roi_height + data->top, 1,
          0, 2 * M_PI);

      cairo_stroke(cr);
    }
  }
}

/**
 * Function handler for sigint signal to stop pipeline
 */
void sigint_handler(int signum) {
  UNUSED(signum);
  if (data.pipeline != nullptr) {
    if (data.source_id != 0) {
      g_print("Stop feeding...\n");
      g_source_remove(data.source_id);
      data.source_id = 0;
    }
    gst_element_send_event(data.pipeline, gst_event_new_eos());
  }
}
