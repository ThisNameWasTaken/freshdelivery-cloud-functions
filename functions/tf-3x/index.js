const functions = require('firebase-functions');
const admin = require('firebase-admin');
const tf = require('@tensorflow/tfjs-node');

const serviceAccount = require('./freshdelivery-586d6-firebase-adminsdk-mjjp1-3b6abae6da.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://freshdelivery-586d6.firebaseio.com',
});

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//   functions.logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });

exports.updateRecommendations = functions.firestore
  .document('reviews/{reviewId}')
  .onCreate(async (snapshot, context) => {
    const tf = require('@tensorflow/tfjs-node');

    const users = [
      'adriandinca2008@gmail.com',
      'adriandinca2008@yahoo.com',
      'adrian-florin.dinca@my.fmi.unibuc.ro',
      'adrian.dinca@s.unibuc.ro',
    ];

    const products = [
      'cow-chese',
      'pineapple',
      'frozen-broccoli',
      'fresh-tomatoes',
      'kiwi',
      'milk',
      'eggs',
      'frozen-smoothie-mix',
      'coca-cola',
      'banana',
      'cheese',
      'orange-juice',
      'green-leaf-lettuce',
    ];

    const features = [
      'dairy',
      'milk',
      'cheese',
      'eggs',
      'fruits',
      'frozen-fruits',
      'fresh-fruits',
      'vegetables',
      'frozen-vegetables',
      'fresh-vegetables',
      'smoothie',
      'drinks',
      'soda',
      'coke',
    ];

    const num_users = users.length;
    const num_products = products.length;
    const num_feats = features.length;
    const num_recommendations = 8;

    const users_products = tf.tensor(
      [
        [8, 7, 0, 0, 8, 7, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 10, 8, 9, 0, 9, 0, 0, 0, 0, 9, 0],
        [0, 0, 0, 8, 0, 0, 0, 8, 7, 0, 0, 10, 9],
        [0, 9, 8, 0, 7, 0, 0, 0, 0, 9, 0, 0, 8],
      ],
      undefined,
      'float32'
    );

    const products_feats = tf.tensor(
      [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
      ],
      undefined,
      'float32'
    );

    let users_feats = tf.matMul(users_products, products_feats);

    users_feats = tf.div(users_feats, tf.sum(users_feats, [1], true));

    // const top_users_features = tf.topk(users_feats, num_feats).indices;

    const users_ratings = tf.matMul(users_feats, tf.transpose(products_feats));

    tf;
    const users_unseen_products = tf.equal(
      users_products,
      tf.zerosLike(users_products)
    );
    const ignore_matrix = tf.zerosLike(tf.cast(users_products, 'float32'));

    const users_ratings_new = tf.where(
      users_unseen_products,
      users_ratings,
      ignore_matrix
    );

    const top_products = tf.topk(users_ratings_new, num_recommendations);
    const top_products_indices = top_products.indices.arraySync();

    const data = {};
    for (let i = 0; i < num_users; i++) {
      data[users[i]] = top_products_indices[i].map((index) => products[index]);
    }

    for (const id in data) {
      admin.firestore().collection('recommendations').doc(id).set(data[id]);
    }
  });
