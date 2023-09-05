const functions = require('firebase-functions');
const admin = require('firebase-admin');
const { Translate } = require('@google-cloud/translate').v2;
require('@tensorflow/tfjs');
const toxicity = require('@tensorflow-models/toxicity');

const serviceAccount = require('./freshdelivery-586d6-firebase-adminsdk-mjjp1-3b6abae6da.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://freshdelivery-586d6.firebaseio.com',
});

const translate = new Translate({
  key: 'AIzaSyAjw59aH8-8wwLb_yNfybAtLFbIMgZDrOM',
});

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//   functions.logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });

exports.removeToxicReviews = functions.firestore
  .document('reviews/{reviewId}')
  .onCreate(async (snapshot, context) => {
    try {
      const review = snapshot.data();
      const { description } = review;

      const [translatedText] = await translate.translate(description, {
        from: 'ro',
        to: 'en',
      });

      const model = await toxicity.load(0.7);
      const predictions = await model.classify(translatedText);
      let isToxic = false;

      predictions.forEach(({ results }) => {
        results.forEach((result) => {
          if (result.match) {
            console.log(
              `Removing review ${snapshot.id}#${description} because it was found to be ${result.label}`
            );
            isToxic = true;
          }
        });
      });

      if (isToxic) {
        await admin.firestore().collection('reviews').doc(snapshot.id).delete();
      }
    } catch (err) {
      console.error(err);
    }
  });
