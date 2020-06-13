<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>Vertica-ML-Python</h1>
        <p>
          Vertica-ML-Python is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Virtual Dataframe), and offers multiple graphical rendering possibilities.
        </p>
      <h1>Advantages</h1>
        <center><ul id="advantages">
          <a href="https://www.vertica.com/"><li>
            <img src="img/VerticaLogo2.png" height="80px">
            <h3><b>Scalable as Vertica</b></h3>
            <div>All the heavy computations are pushed to Vertica. Get advantages of the columnar MPP infrastructure built for analytics. Compressions and projections are key elements to high performance.</div>
          </li></a>
          <a href="about.php"><li>
            <img src="img/pythonLogo.png" height="80px">
            <h3><b>Flexible as Python</b></h3>
            <div>The API is entirely coded using Python. It generates SQL code needed to fulfill the process. Implementing new customized functionnalities become easy. You'll be able to write less code and do more.</div>
          </li></a>
          <a href="documentation_last/vdataframe/index.php"><li>
            <img src="img/pandasLogo.png" height="80px">
            <h3><b>Pandas like Functionalities</b></h3>
            <div>Uses the advantage of a new object: the vDataFrame which is very similar to pandas DF in the use but push back all the heavy computations to Vertica. vDataFrame is the natural transition from Small to Big Data.</div>
          </li></a>
          <a href="documentation_last/index.php"><li>
            <img src="img/inDB.png" height="80px">
            <h3><b>In-Database Data Science</b></h3>
            <div>Vertica ML Python is using Vertica in-DB ML and advanced SQL built-in functionnalities to propose plenty of abstractions to simplify the entire Data Science Process. Bring the Logic to the data and not the opposite.</div>
          </li></a>
        </ul></center>
      <h1>Features</h1>
      <p>
        Vertica ML Python is the perfect combination between Vertica and Python. It uses Vertica Scalability and Python Flexibility to help any Data Scientist achieving his goals by bringing the logic to the data and not the opposite. With VERTICA ML PYTHON, start your journey with easy Data Exploration. 
      </p>
        <br><center><ul id="advantages">
          <a href="documentation_last/vdataframe/statistics.php"><li>
            <img src="img/icons/Budget.png" height="180px">
            <h3><b>Descriptive Statistics</b></h3>
            <div>Explore all Statistics</div>
            <i class="small">Describes, Aggregates...</i>
          </li></a>
          <a href="documentation_last/vdataframe/plot.php"><li>
            <img src="img/icons/Pie Chart.png" height="180px">
            <h3><b>Charts</b></h3>
            <div>Vizualize your Data</div>
            <i class="small">Histograms, Pies, Hexbins...</i>
          </li></a>
          <a href="documentation_last/vdataframe/corr.php"><li>
            <img src="img/icons/Co-Working.png" height="180px">
            <h3><b>Correlation & Dependancy</b></h3>
            <div>Find Variables Links</div>
            <i class="small">Correlation Matrix, Autocorrelation Plot...</i>
          </li></a>
        </ul></center><br>
      <p>
        Prepare your Data and do advanced Features Engineering using Advanced Analytical Functions and Moving Windows.
      </p>
        <br><center><ul id="advantages">
          <a href="documentation_last/vdataframe/features.php"><li>
            <img src="img/icons/Fresh Idea.png" height="180px">
            <h3><b>Features Engineering</b></h3>
            <div>Advanced Analytical Functions</div>
            <i class="small">Sessionization, Moving Windows...</i>
          </li></a>
          <a href="documentation_last/vdataframe/preprocessing.php"><li>
            <img src="img/icons/SEO Content Management.png" height="180px">
            <h3><b>Data Cleaning</b></h3>
            <div>Smart Preprocessing</div>
            <i class="small">Normalization, Encoding...</i>
          </li></a>
        </ul></center><br>
      <p>
        Create a model with Highly Scalable Vertica ML. Evaluate your model and try to create the most efficient and performant one. Many in-DataBase Machine Learning Algorithms are available built to scale.
      </p>
      <br><center><ul id="advantages">
          <a href="documentation_last/learn/examples/regression/index.php"><li>
            <img src="img/icons/Analysis.png" height="180px">
            <h3><b>Regression</b></h3>
            <div>Forecast sales revenues.</div>
            <i class="small">Linear Regression, KNN...</i>
          </li></a>
          <a href="documentation_last/learn/examples/binary-classification/index.php"><li>
            <img src="img/icons/Teamwork Problem Fix.png" height="180px">
            <h3><b>Binary Classification</b></h3>
            <div>Predict sensor failure.</div>
            <i class="small">Logistic Regression, SVM...</i>
          </li></a>
          <a href="documentation_last/learn/examples/unsupervised/index.php"><li>
            <img src="img/icons/Conference.png" height="180px">
            <h3><b>Clustering</b></h3>
            <div>Customer segmentation.</div>
            <i class="small">KMeans, DBSCAN...</i>
          </li></a>
          <a href="documentation_last/learn/examples/multiclass-classification/index.php"><li>
            <img src="img/icons/Business Hierarcy.png" height="180px">
            <h3><b>Multiclass Classification</b></h3>
            <div>Classify gene expression data.</div>
            <i class="small">Naive Bayes, Random Forest...</i>
          </li></a>
          <a href="documentation_last/learn/examples/decomposition/index.php"><li>
            <img src="img/icons/Management.png" height="180px">
            <h3><b>Decomposition</b></h3>
            <div>Keep only Important Information.</div>
            <i class="small">PCA, SVD...</i>
          </li></a>
        </ul></center><br>
      <p>
        Everything will happen in one place and where it should be: <b>your Database</b>.
      </p>
      <div><?php include('include/footer.php'); ?></div>
      </div>
    </body>
</html>