import { StepType } from '@reactour/tour'

export const tutorialSteps: StepType[] = [
    { // 0
        selector: 'body',
        content: 'Welcome to APalysis. A tool to analyze the activation behaviors of Neural Networks using Data at Real-Time.',
    },
    { // 1
        selector: '.tutorial-tutorial',
        content: 'You can click on the "Tutorial" button anytime to see this tutorial again.',
    },
    { // 2
        selector: 'body',
        content: 'The whole visualization is divided into to parts: the main view and the controls.',
    },
    { // 3
        selector: '.tutorial-control',
        content: 'This panel has all the controls to configure the the model is fed with. ',
    },
    { // 4
        selector: '.tutorial-control',
        content: 'The loaded neural network will be run with a number of sample inputs from the dataset for quick-look. For a classification task, you can select the classes you want to analyze the model for, and the number of samples per class.',
    },
    { // 5
        selector: '.tutorial-image-per-class',
        content: 'This lets you select the number of samples per class to visualize.',
    },
    { // 6
        selector: '.tutorial-shuffle',
        content: 'This lets you choose whether you want to shuffle from the whole dataset, or pick the first n images from the dataset of each class (faster).',
    },
    { // 7
        selector: '.tutorial-selected-classes',
        content: 'Select the classes you want to visualize. It is recommended that you select at most 4 classes for better visualization.',
    },
    { // 8
        selector: '.tutorial-analyze',
        content: 'Now click the analyze button to run the model with the selected configuration. This may take some time depending on the model, the number of samples you selected and the computer you are running this on.',
    },
    { // 9
        selector: '.tutorial-input-images',
        content: 'These are the sample images that the model was run with. The bottom text of each image is the prediction of the model, green being the correct prediction and red being the wrong prediction.',
    },
    { // 10
        selector: '.tutorial-main-view',
        content: 'Here is the main view of the visualization. The neural network is shown as a topologically sorted graph. Each node represents a layer in the neural network. You can zoom, pan and drag the graph around.',
    },
    { // 11
        selector: '.tutorial-main-view-controls',
        content: 'For better accessibility, the zoom functionality, drag-n-drop lock buttons are here. The "H" means the graph is showed in horizontal orientation, and the "V" (after clicking "H") means the graph is showed in vertical orientation. Finally, you can export the graph as an image by clicking the Export button at the end.',
    },
    { // 12
        selector: '.tutorial-cnn-layer',
        content: 'Lets zoom into a CNN layer of the network to get some details of that layers activations for the selected input images.',
    },
    { // 13
        selector: '.tutorial-cnn-layer',
        content: 'Each node has a number of accordion items. You can click on the header to expand/collapse the accordion item.',
    },
    { // 14
        selector: '.tutorial-cnn-layer-details',
        content: 'The first accordion item shows some details about the layer.',
    },
    { // 15
        selector: '.tutorial-cnn-layer-heatmap',
        content: 'The next one is the activation heatmap. To learn more about the heatmap, please see the related paper. ',
    },
    { // 16
        selector: '.tutorial-cnn-layer',
        content: 'Finally there are also a bunch of other accordions that show other details about the layer. Each are explained in the paper. The user can also add their own accordions to show custom details about the layer using the APalysis custom accordion API.',
    },
    { // 17
        selector: 'body',
        content: 'Thank you for taking the time to go through this tutorial. You can click on the "Tutorial" button anytime to see this tutorial again.',
    },
]