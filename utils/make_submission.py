from submission import masks_to_submission

submission_filename = './predictions/bin_eq++_DBSCAN/bin_uneq1_500_0.4.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = './predictions/bin_eq++_DBSCAN/pred_test_' + str(i) + '.png'
    print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)