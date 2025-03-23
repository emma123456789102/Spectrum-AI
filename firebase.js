// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBqlDdqZXwBANTYLOuuTYDIB7U4SCdKvGc",
  authDomain: "spectrumai-f2aad.firebaseapp.com",
  projectId: "spectrumai-f2aad",
  storageBucket: "spectrumai-f2aad.firebasestorage.app",
  messagingSenderId: "916233718480",
  appId: "1:916233718480:web:d14e87eb7c22a23b252cd5",
  measurementId: "G-ZQSRRPLZBW"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
