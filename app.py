from flask import Flask,render_template,request,send_file
import matplotlib.pyplot as plt
import numpy as np
from main import *
from io import BytesIO


app=Flask(__name__)
@app.route('/home')
def home():
    return render_template(
        
        'index.html',
        title="Home",
    )
    
from PIL import Image
@app.route('/home1',methods=['POST'])
def home1():
    file=request.files['image']
    original_img = plt.imread(file) / 255.0  # Normalize to [0, 1]
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
    K = 16
    max_iters = 10
    initial_centroids = kMeans_init_centroids(X_img, K)
    print("Initial centroids:", initial_centroids)
    centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
    print("Centroids after K-means:", centroids)
    idx = find_closest_centroids(X_img, centroids)
    X_recovered = centroids[idx, :] 
    X_recovered = np.reshape(X_recovered, original_img.shape) 
    print("X_recovered shape:", X_recovered.shape)
    
    #X_recovered = np.clip(X_recovered, 0, 1)
    
    print("X_recovered min:", X_recovered.min(), "max:", X_recovered.max())
    compressed_image = Image.fromarray((X_recovered * 255).astype(np.uint8),'RGB')
    
    
    # Save the figure to a BytesIO object
    output = BytesIO()
    compressed_image.save(output, format='JPEG')  
    #fig.savefig(output, format='JPEG')
    #plt.close(fig)
    output.seek(0)
    
    return send_file(output, mimetype='image/jpeg',as_attachment=True,
        download_name="compressed_image.jpg")
    

@app.route('/')
def start():
    return render_template(
        
        'start.html',
        title="Start",
    )
@app.route('/about')
def aboutus():
    return render_template(
        
        'about.html',
        title="What we do"
    )
    
    
    






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)