// Package yolov5 provides a Go implementation of the YOLO V5 object detection system: https://pjreddie.com/darknet/yolo/.
//
// The yolov5 package leverages gocv(https://github.com/hybridgroup/gocv) for a neural net able to detect object.
//
// In order for the neural net to be able to detect objects, it needs the pre-trained network model
// consisting of a .cfg file and a .weights file. Using the Makefile provied by the library, these models
// can simply be downloaded by running 'make models'.
//
// In order to use the package, make sure you've checked the prerequisites in the README: https://github.com/wimspaargaren/yolov5#prerequisites
package yolov5

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"

	"gocv.io/x/gocv"

	"github.com/sirno-baka/yolov5/internal/ml"
)

// Default constants for initialising the yolov5 net.
const (
	DefaultRows        = 25200
	DefaultStepSize    = 85
	DefaultYoloVersion = 5
	DefaultInputWidth  = 1280
	DefaultInputHeight = 1280

	DefaultConfThreshold float32 = 0.5
	DefaultNMSThreshold  float32 = 0.4
)

// Config can be used to customise the settings of the neural network used for object detection.
type Config struct {
	Rows     int
	StepSize int
	// InputWidth & InputHeight are used to determine the input size of the image for the network
	InputWidth  int
	InputHeight int
	YoloVersion int
	// ConfidenceThreshold can be used to determine the minimum confidence before an object is considered to be "detected"
	ConfidenceThreshold float32
	// Non-maximum suppression threshold used for removing overlapping bounding boxes
	NMSThreshold float32

	// Type on which the network will be executed
	NetTargetType  gocv.NetTargetType
	NetBackendType gocv.NetBackendType

	// NewNet function can be used to inject a custom neural net
	NewNet func(modelPath string) ml.NeuralNet
}

// validate ensures that the basic fields of the config are set
func (c *Config) validate() {
	if c.NewNet == nil {
		c.NewNet = initializeNet
	}
	if c.InputWidth == 0 {
		c.InputWidth = DefaultInputWidth
	}
	if c.InputHeight == 0 {
		c.InputHeight = DefaultInputHeight
	}
	if c.YoloVersion == 0 {
		c.YoloVersion = DefaultYoloVersion
	}
}

// DefaultConfig used to create a working yolov5 net out of the box.
func DefaultConfig() Config {
	return Config{
		Rows:                DefaultRows,
		StepSize:            DefaultStepSize,
		InputWidth:          DefaultInputWidth,
		InputHeight:         DefaultInputHeight,
		ConfidenceThreshold: DefaultConfThreshold,
		NMSThreshold:        DefaultNMSThreshold,
		NetTargetType:       gocv.NetTargetCPU,
		NetBackendType:      gocv.NetBackendDefault,
		NewNet:              initializeNet,
	}
}

// ObjectDetection represents information of an object detected by the neural net.
type ObjectDetection struct {
	ClassID     int
	ClassName   string
	BoundingBox image.Rectangle
	Confidence  float32
}

// Net the yolov5 net.
type Net interface {
	Close() error
	GetDetections(gocv.Mat) ([]ObjectDetection, error)
	GetDetectionsWithFilter(gocv.Mat, map[string]bool) ([]ObjectDetection, error)
}

// yoloNet the net implementation.
type yoloNet struct {
	net       ml.NeuralNet
	cocoNames []string

	Rows                int
	StepSize            int
	YoloVersion         int
	DefaultInputWidth   int
	DefaultInputHeight  int
	confidenceThreshold float32
	DefaultNMSThreshold float32
}

// NewNet creates new yolo net for given weight path, config and coconames list.
func NewNet(modelPath, cocoNamePath string) (Net, error) {
	return NewNetWithConfig(modelPath, cocoNamePath, DefaultConfig())
}

// NewNetWithConfig creates new yolo net with given config.
func NewNetWithConfig(modelPath, cocoNamePath string, config Config) (Net, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("path to net model not found")
	}

	cocoNames, err := getCocoNames(cocoNamePath)
	if err != nil {
		return nil, err
	}

	config.validate()

	net := config.NewNet(modelPath)

	err = setNetTargetTypes(net, config)
	if err != nil {
		return nil, err
	}

	return &yoloNet{
		net:                 net,
		Rows:                config.Rows,
		StepSize:            config.StepSize,
		cocoNames:           cocoNames,
		YoloVersion:         config.YoloVersion,
		DefaultInputWidth:   config.InputWidth,
		DefaultInputHeight:  config.InputHeight,
		confidenceThreshold: config.ConfidenceThreshold,
		DefaultNMSThreshold: config.NMSThreshold,
	}, nil
}

// initializeNet default method for creating neural network, leveraging gocv.
func initializeNet(modelPath string) ml.NeuralNet {
	net := gocv.ReadNetFromONNX(modelPath)
	return &net
}

func setNetTargetTypes(net ml.NeuralNet, config Config) error {
	err := net.SetPreferableBackend(config.NetBackendType)
	if err != nil {
		return err
	}

	err = net.SetPreferableTarget(config.NetTargetType)
	if err != nil {
		return err
	}
	return nil
}

// Close closes the net.
func (y *yoloNet) Close() error {
	return y.net.Close()
}

// GetDetections retrieve predicted detections from given matrix.
func (y *yoloNet) GetDetections(frame gocv.Mat) ([]ObjectDetection, error) {
	return y.GetDetectionsWithFilter(frame, make(map[string]bool))
}

// GetDetectionsWithFilter allows you to detect objects, but filter out a given list of coco name ids.
func (y *yoloNet) GetDetectionsWithFilter(frame gocv.Mat, classIDsFilter map[string]bool) ([]ObjectDetection, error) {
	blob := gocv.BlobFromImage(frame, 1.0/255.0, image.Pt(y.DefaultInputWidth, y.DefaultInputHeight), gocv.NewScalar(0, 0, 0, 0), true, false)
	// nolint: errcheck
	defer blob.Close()
	y.net.SetInput(blob, "")
	layerIDs := y.net.GetUnconnectedOutLayers()
	fl := []string{}

	for _, id := range layerIDs {
		layer := y.net.GetLayer(id)
		fl = append(fl, layer.GetName())
	}
	outputs := y.net.ForwardLayers(fl)
	for i := 0; i < len(outputs); i++ {
		// nolint: errcheck
		defer outputs[i].Close()
	}

	detections, err := y.processOutputs(frame, outputs, classIDsFilter)
	if err != nil {
		return nil, err
	}

	return detections, nil
}

// processOutputs process detected rows in the outputs.
func (y *yoloNet) processOutputs(frame gocv.Mat, outputs []gocv.Mat, filter map[string]bool) ([]ObjectDetection, error) {
	// FIXME add filter functionality
	_ = filter

	detections := []ObjectDetection{}
	bboxes := []image.Rectangle{}
	confidences := []float32{}

	rows := outputs[0].Size()[1]
	dimensions := outputs[0].Size()[2]
	yolov8 := y.YoloVersion == 8
	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if dimensions > rows {
		rows = outputs[0].Size()[2]
		dimensions = outputs[0].Size()[1]
		outputs[0] = outputs[0].Reshape(1, dimensions)
		gocv.Transpose(outputs[0], &outputs[0])
		yolov8 = true
	}

	data, err := outputs[0].DataPtrFloat32()
	if err != nil {
		return nil, err
	}

	for i := 0; i < rows; i++ {

		if yolov8 {
			startIndex := 4 + dimensions*i
			endIndex := dimensions * (i + 1)
			scores := data[startIndex:endIndex]
			classID, confidence := getClassID(scores)

			// Print the results
			if confidence > y.confidenceThreshold {
				//fmt.Printf("Max class score: %f, Class ID: %d\n", maxClassScore, classID)
				confidences = append(confidences, confidence)
				boundingBox := y.calculateBoundingBox(frame, data[0+dimensions*i:4+dimensions*i])
				bboxes = append(bboxes, boundingBox)
				detections = append(detections, ObjectDetection{
					ClassID:     classID,
					ClassName:   y.cocoNames[classID],
					BoundingBox: boundingBox,
					Confidence:  confidence,
				})
			}
		} else {
			confidence := data[4+dimensions*i]
			if confidence >= y.confidenceThreshold {
				startIndex := 5 + dimensions*i
				endIndex := dimensions * (i + 1)
				scores := data[startIndex:endIndex]
				classID, _ := getClassID(scores)
				confidences = append(confidences, confidence)
				boundingBox := y.calculateBoundingBox(frame, data[0+dimensions*i:4+dimensions*i])
				bboxes = append(bboxes, boundingBox)
				detections = append(detections, ObjectDetection{
					ClassID:     classID,
					ClassName:   y.cocoNames[classID],
					BoundingBox: boundingBox,
					Confidence:  confidence,
				})
			}
		}
	}

	if len(bboxes) == 0 {
		return detections, nil
	}
	indices := make([]int, len(bboxes))

	gocv.NMSBoxes(bboxes, confidences, y.confidenceThreshold, y.DefaultNMSThreshold, indices)
	result := []ObjectDetection{}
	for i, indice := range indices {
		// If we encounter value 0 skip the detection
		// except for the first indice
		if i != 0 && indice == 0 {
			continue
		}
		result = append(result, detections[indice])
	}
	return result, nil
}

func (y *yoloNet) isFiltered(classID int, classIDs map[string]bool) bool {
	if classIDs == nil {
		return false
	}
	return classIDs[y.cocoNames[classID]]
}

// calculateBoundingBox calculate the bounding box of the detected object.
func (y *yoloNet) calculateBoundingBox(frame gocv.Mat, row []float32) image.Rectangle {
	if len(row) < 4 {
		return image.Rect(0, 0, 0, 0)
	}

	xFactor := float32(frame.Cols()) / float32(y.DefaultInputWidth)
	yFactor := float32(frame.Rows()) / float32(y.DefaultInputHeight)

	x, y1, w, h := row[0], row[1], row[2], row[3]
	left := int((x - 0.5*w) * xFactor)
	top := int((y1 - 0.5*h) * yFactor)
	width := int(w * xFactor)
	height := int(h * yFactor)

	return image.Rect(left, top, left+width, top+height)
}

func getClassID(x []float32) (int, float32) {
	res := 0
	max := float32(0)
	for i, y := range x {
		if y > max {
			res = i
			max = y
		}
	}
	return res, max
}

// getCocoNames read coconames from given path.
func getCocoNames(path string) ([]string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return strings.Split(string(content), "\n"), nil
}

// DrawDetections draws a given list of object detections on a gocv Matrix.
func DrawDetections(frame *gocv.Mat, detections []ObjectDetection) {
	for i := 0; i < len(detections); i++ {
		detection := detections[i]
		text := fmt.Sprintf("%s:%.2f%%", detection.ClassName, detection.Confidence*100)

		// Create bounding box of object
		blue := color.RGBA{0, 0, 255, 0}
		gocv.Rectangle(frame, detection.BoundingBox, blue, 3)

		// Add text background
		black := color.RGBA{0, 0, 0, 0}
		size := gocv.GetTextSize(text, gocv.FontHersheySimplex, 0.5, 1)
		r := detection.BoundingBox
		textBackground := image.Rect(r.Min.X, r.Min.Y-size.Y-1, r.Min.X+size.X, r.Min.Y)
		gocv.Rectangle(frame, textBackground, black, int(gocv.Filled))

		// Add text
		pt := image.Pt(r.Min.X, r.Min.Y-4)
		white := color.RGBA{255, 255, 255, 0}
		gocv.PutText(frame, text, pt, gocv.FontHersheySimplex, 0.5, white, 1)
	}
}
