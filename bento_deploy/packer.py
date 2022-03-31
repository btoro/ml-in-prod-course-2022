from bento_service import HFClassifier
import pickle 

# Create a iris classifier service instance
hf_classifier_service = HFClassifier()


with open("models/HF.pkl",'rb') as f:
    clf = pickle.load(f)
# Pack the newly trained model artifact
hf_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = hf_classifier_service.save()