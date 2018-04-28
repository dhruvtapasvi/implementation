function generateUserStudy() {
    var USER_STUDY_TITLE = "User Study Survey";
    var WELCOME_TEXT = "Thank you for agreeing to participate in my user study!";
    var CONSENT_TEXT = "Before beginning the survey, please read the following consent constructions.";
    var CONSENT_AGREEMENT_TEXT = "I have read the above instructions and consent to carry out the user study.";
    var INSTRUCTIONS_TEXT = "Some Instructions";
    var QUESTIONS_TEXT = "Please answer ALL questions.";
    var RATE_RECONSTRUCTION_QUESTION = "Please rate the reconstruction out of 5:";
    var RATE_INTERPOLATION_QUESTION = "Please rate the interpolation out of 5:";
    var RECONSTRUCTION_OPTIONS = [
        "1: ",
        "2: ",
        "3: ",
        "4: ",
        "5: "
    ];
    var INTERPOLATION_OPTIONS = [
        "1: ",
        "2: ",
        "3: ",
        "4: ",
        "5: "
    ];
    var IMAGES_FOLDER = DriveApp.getFolderById('1MYrga5FwBAswPJRu0HC-9ATPURqIltFC');

    var PARTICIPANT_NUMBER = 0;
    var NUM_RECONSTRUCTION_QUESTIONS = 32;
    var NUM_INTERPOLATION_QUESTIONS = 48;

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
