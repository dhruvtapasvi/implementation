function generateUserStudy() {
    var PARTICIPANT_NUMBER = 0;
    var USER_STUDY_TITLE = "Survey (#".concat(PARTICIPANT_NUMBER.toString(), ")");
    var WELCOME_TEXT = "Thank you for agreeing to participate in my survey!";

    var CONSENT_TEXT = "Please read the following instructions regarding consent before beginning the survey:\n" +
        "Participation in this survey is entirely voluntary. You may freely choose not to carry out this survey without justification or penalty.\n" +
        "Additionally, even after consenting to carry out the study, you may withdraw your consent at any time and stop doing the survey.\n" +
        "At any time, if you wish to stop doing the survey, please close this browser window, and DO NOT submit answers or proceed further with the survey.\n" +
        "This survey is designed to about 10 minutes. If at any stage your feel overstretched while carrying out the survey, " +
        "or if the survey is taking a disproportionate amount of time (more than about 30 minutes) to complete, " +
        "please feel free to withdraw from the survey.\n" +
        "No private data will be collected during the course of the survey. The survey is entirely anonymous, and data collected will only relate to rating images.\n" +
        "Please contact me by email at \"ddt21@cam.ac.uk\" if you have any further questions about this survey.";
    var CONSENT_AGREEMENT_TEXT = "I have read and understood the above instructions, and consent to carry out the user study.";

    var INSTRUCTIONS_TEXT = "Please take the time to read these additional instructions before proceeding with the survey:\n" +
        "There will be two types of questions in the survey. Both will involve carrying out a rating of scores between 1 and 5 inclusive. The explanations for each scoring category are specified in each question.\n" +
        "The first type of question will ask you to rate a \"reconstruction\". There will be two pictures, side-by-side. The aim is to rate how well the second image (on the right) resembles the first image (on the left).\n" +
        "The second type of question will ask you to rate an \"interpolation\". There will be three pictures arranged side-by-side in a horizontal line. The aim is to rate how well the middle image fits in a sequence between the image to the left and the image to the right.";

    var QUESTIONS_TEXT = "Please answer ALL questions.\n" +
        "RECONSTRUCTION questions: There will be two pictures, side-by-side. The aim is to rate how well the second image (on the right) resembles the first image (on the left).\n" +
        "INTERPOLATION questions: There will be three pictures arranged side-by-side in a horizontal line. The aim is to rate how well the middle image fits in a sequence between the image to the left and the image to the right.";
    var RATE_RECONSTRUCTION_QUESTION = "Please rate the reconstruction out of 5:";
    var RATE_INTERPOLATION_QUESTION = "Please rate the interpolation out of 5:";
    var RECONSTRUCTION_OPTIONS = [
        "1: The second image does not resemble the first in any way.",
        "2: The second image only vaguely resembles the first, e.g. has approximately the correct shape.",
        "3: The second image somewhat resembles the first, possibly with faults of differences.",
        "4: The second image clearly resembles the first, with some minor faults or differences.",
        "5: The second image perfectly resembles the first."
    ];
    var INTERPOLATION_OPTIONS = [
        "1: The central image does not form any distinguishable sequence between the first and the last.",
        "2: The central image forms a barely distinguishable sequence between the first and the last.",
        "3: The central image forms an somewhat distinguishable sequence between the first and the last.",
        "4: The central image forms a clearly distinguishable sequence between the first and the last.",
        "5: The central image forms a perfect sequence between the first and the last."
    ];
    var IMAGES_FOLDER = DriveApp.getFolderById('1LxmUULjAXlIeNyi4iBKmld6as5z2Q-qi');

    var NUM_RECONSTRUCTION_QUESTIONS = 16;
    var NUM_INTERPOLATION_QUESTIONS = 24;

    var START_INDEX_RECONSTRUCTIONS = PARTICIPANT_NUMBER * NUM_RECONSTRUCTION_QUESTIONS;
    var END_INDEX_RECONSTRUCTIONS = START_INDEX_RECONSTRUCTIONS + NUM_RECONSTRUCTION_QUESTIONS;
    var START_INDEX_INTERPOLATIONS = PARTICIPANT_NUMBER * NUM_INTERPOLATION_QUESTIONS;
    var END_INDEX_INTERPOLATIONS = START_INDEX_INTERPOLATIONS + NUM_INTERPOLATION_QUESTIONS;

    var form = FormApp.create(USER_STUDY_TITLE);

    var greetingSection = form.addSectionHeaderItem();
    greetingSection.setTitle("Introduction");
    greetingSection.setHelpText(WELCOME_TEXT);

    var consentCheckbox = form.addCheckboxItem();
    consentCheckbox.setTitle("Consent");
    consentCheckbox.setHelpText(CONSENT_TEXT);
    consentCheckbox.setChoices([
        consentCheckbox.createChoice(CONSENT_AGREEMENT_TEXT)
    ]);

    var consentCheckboxValidation = FormApp.createCheckboxValidation().requireSelectExactly(1).build();
    consentCheckbox.setValidation(consentCheckboxValidation);

    pageTwo = form.addPageBreakItem();
    pageTwo.setTitle("Instructions");
    pageTwo.setHelpText(INSTRUCTIONS_TEXT);


    pageThree = form.addPageBreakItem();
    pageThree.setTitle("Questions");
    pageThree.setHelpText(QUESTIONS_TEXT);

    var subFolders = IMAGES_FOLDER.getFolders();
    while (subFolders.hasNext()) {
        var subFolder = subFolders.next();
        generateImageQuestions(form, subFolder, "reconstruction", RATE_RECONSTRUCTION_QUESTION, RECONSTRUCTION_OPTIONS, START_INDEX_RECONSTRUCTIONS, END_INDEX_RECONSTRUCTIONS);
        generateImageQuestions(form, subFolder, "interpolation", RATE_INTERPOLATION_QUESTION, INTERPOLATION_OPTIONS, START_INDEX_INTERPOLATIONS, END_INDEX_INTERPOLATIONS);
    }
}

function generateImageQuestions(form, subFolder, subSubFolderName, questionText, answerOptions, startIndex, endIndex) {
    var reconstructionFolder = subFolder.getFoldersByName(subSubFolderName).next();
    var i;
    for (i = startIndex; i < endIndex; i++) {
        var fileName = "image_".concat(i.toString(), ".png");
        var fileWithName = reconstructionFolder.getFilesByName(fileName).next();
        var fileContent = fileWithName.getBlob();

        var subFolderName = subFolder.getName();
        var questionName = subFolderName.concat("_", subSubFolderName, "_", i.toString());

        var image = form.addImageItem();
        image.setTitle(questionText);
        image.setImage(fileContent);

        var question = form.addMultipleChoiceItem();
        question.setTitle(questionName);
        question.setChoices(answerOptions.map(function (option) {
            return question.createChoice(option);
        }));
        question.setRequired(true);
    }
}
