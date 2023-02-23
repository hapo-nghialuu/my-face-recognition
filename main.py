import face_recognition


# Example usage face_recognition
def example():
    image = face_recognition.load_image_file("images/my_picture.jpg")
    face_locations = face_recognition.face_locations(image)
    face_landmarks = face_recognition.face_landmarks(image)

    print(face_locations)
    print('\n')
    print(face_landmarks)
    return


# Recognize faces in images and identify who they are
def identify_faces():
    picture_of_me = face_recognition.load_image_file("images/me.png")
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
    unknown_picture = face_recognition.load_image_file("images/unknown.jpg")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)

    # Now we can see the two face encodings are of the same person with `compare_faces`!
    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print("It's a picture of me!")
    else:
        print("It's not a picture of me!")

    return


if __name__ == '__main__':
    identify_faces()
