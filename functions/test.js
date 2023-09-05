(async () => {
  const admin = require('firebase-admin');

  const serviceAccount = require('./freshdelivery-586d6-firebase-adminsdk-mjjp1-3b6abae6da.json');

  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: 'https://freshdelivery-586d6.firebaseio.com',
  });

  require('@tensorflow/tfjs3');
  const tf = require('@tensorflow/tfjs-node3');

  const { users } = await admin.auth().listUsers();

  const productsData = (
    await admin.firestore().collection('products').get()
  ).docs.map((doc) => doc.data());
  const features = [];
  const products = [];

  for (const product of productsData) {
    products.push(product.slug);

    for (const tag of product.tag) {
      if (features.indexOf(tag.slug) === -1) {
        features.push(tag.slug);
        if (tag.slug === undefined || tag.slug === 'Drinks') {
          console.log(product);
        }
      }
    }
  }

  const recommendationsCount = 8;

  const reviews = (
    await admin.firestore().collection('reviews').get()
  ).docs.map((doc) => doc.data());

  const emptyRatingsRaw = new Array(products.length).fill(0);
  const userRatingsRaw = new Array(users.length)
    .fill(0)
    .map((el) => [...emptyRatingsRaw]);

  for (const review of reviews) {
    const userIndex = users.findIndex((user) => user.uid === review.userId);
    const productIndex = products.findIndex(
      (product) => product === review.productSlug
    );

    userRatingsRaw[userIndex][productIndex] = review.rating;
  }

  const userRatings = tf.tensor(userRatingsRaw, undefined, 'float32');

  const emptyFeaturesRaw = new Array(features.length).fill(0);
  const productFeaturesRaw = new Array(products.length)
    .fill(0)
    .map((el) => [...emptyFeaturesRaw]);

  for (let i = 0; i < productsData.length; i++) {
    const product = productsData[i];
    for (const tag of product.tag) {
      const featureIndex = features.indexOf(tag.slug);
      productFeaturesRaw[i][featureIndex] = 1;
    }
  }

  const productFeatures = tf.tensor(productFeaturesRaw, undefined, 'float32');

  let usersFeatures = tf.matMul(userRatings, productFeatures);
  usersFeatures = tf.div(usersFeatures, tf.sum(usersFeatures, [1], true));

  const usersRatings = tf.matMul(usersFeatures, tf.transpose(productFeatures));

  const usersUnseenProducts = tf.equal(userRatings, tf.zerosLike(userRatings));
  const ignoreMatrix = tf.zerosLike(tf.cast(userRatings, 'float32'));

  const usersRatingsMasked = tf.where(
    usersUnseenProducts,
    usersRatings,
    ignoreMatrix
  );

  const topProducts = tf.topk(usersRatingsMasked, recommendationsCount);
  const topProductsIndices = topProducts.indices.arraySync();

  const data = {};
  for (let i = 0; i < users.length; i++) {
    data[users[i].uid] = topProductsIndices[i].map((index) => products[index]);
  }

  console.log(data);
})();
