// social-media-linear-models


// Define social media data
const socialMediaData = [
    { platform_count: "Facebook, Twitter", avg_time: "Between 2 and 3 hours", age: 25, gender: "Male", self_image_1: 3, self_image_2: 4, self_image_3: 5, self_image_4: 4 },
    { platform_count: "Instagram", avg_time: "Between 3 and 4 hours", age: 30, gender: "Female", self_image_1: 4, self_image_2: 3, self_image_3: 5, self_image_4: 4 },
    // Add more data rows as needed
];

// Define world data (sample data)
const worldData = [
    { platform_count: "Twitter", avg_time: "Between 1 and 2 hours", age: 35, gender: "Male" },
    { platform_count: "Instagram, Facebook", avg_time: "Between 4 and 5 hours", age: 20, gender: "Female" },
    // Add more data rows as needed
];

// Function to count elements in a comma-separated string
function countElements(x) {
    if (x === null || x === undefined) {
        return 0;
    } else {
        const elements = x.split(",").filter(element => element.trim() !== "");
        return elements.length;
    }
}

// Function to convert time string to numeric
function timeToNum(x) {
    switch (x) {
        case "Less than an Hour":
            return 0.5;
        case "Between 1 and 2 hours":
            return 1.5;
        case "Between 2 and 3 hours":
            return 2.5;
        case "Between 3 and 4 hours":
            return 3.5;
        case "Between 4 and 5 hours":
            return 4.5;
        default:
            return 5.5;
    }
}

// Function to convert gender string to numeric
function genderToNum(x) {
    switch (x) {
        case "Male":
            return 0;
        case "Female":
            return 1;
        default:
            return 2;
    }
}

// Manipulate data
socialMediaData.forEach(row => {
    row.platform_count = countElements(row.platform_count);
    row.avg_time = timeToNum(row.avg_time);
    row.gender = genderToNum(row.gender);
});

// MODEL BUILDING --------------------------------------------------------------

// Self Image Model
const selfImageModel = new ML.MultivariateLinearRegression(
    socialMediaData.map(row => [row.platform_count, row.avg_time, row.age, row.gender]),
    socialMediaData.map(row => [row.self_image_1, row.self_image_2, row.self_image_3, row.self_image_4])
);

// Addictive Model
const addictiveModel = new ML.MultivariateLinearRegression(
    socialMediaData.map(row => [row.platform_count, row.avg_time, row.age, row.gender]),
    socialMediaData.map(row => [row.addictive_1, row.addictive_2])
);

// Procrastination Model
const procrastinationModel = new ML.MultivariateLinearRegression(
    socialMediaData.map(row => [row.platform_count, row.avg_time, row.age, row.gender]),
    socialMediaData.map(row => [row.procrasinate_1, row.procrasinate_2, row.procrasinate_3, row.procrasinate_4])
);

// -----------------------------------------------------------------------------

// Predictions for new data
const predictions = worldData.map(row => {
    const prediction = selfImageModel.predict([[countElements(row.platform_count), timeToNum(row.avg_time), row.age, genderToNum(row.gender)]]);
    return prediction[0];
});

console.log("Predictions", predictions);
