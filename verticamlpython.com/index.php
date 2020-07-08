<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>VerticaPy</h1>
        <p>
          Vertica-ML-Python is a Python library that exposes scikit-like functionality to conduct data science projects on data stored in Vertica, taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operations (called Virtual Dataframe), and offers several options for graphical rendering.
        </p>
      <h1>Advantages</h1>
        <center><ul id="advantages_main">
          <li>
            <img src="img/VerticaLogo2.png">
            <h3>Vertica Scalability</h3>
            <div class="description">All heavy computations are pushed to Vertica. Get advantages of a columnar MPP infrastructure built for analytics.</div>
            <center><a href="https://www.vertica.com/secrets-behind-verticas-performance/"><div class="button">Learn More</div></a></center>
          </li>
          <li>
            <img src="img/pythonLogo.png">
            <h3>Python Flexibility</h3>
            <div class="description">The API is coded entirely in Python and generates SQL code on-the-fly to interact with your database.</div>
            <center><a href="http://www.allaboutweb.biz/how-flexible-is-python/"><div class="button">Learn More</div></a></center>
          </li><br>
          <li>
            <img src="img/pandasLogo.png">
            <h3>Pandas like Functionalities</h3>
            <div class="description">The vDataFrame object provides functionality similar to the DataFrame in pandas, but pushes all the heavy computations to Vertica, allowing a seamless transition from small to big data.</div>
            <center><a href="./workshop/introduction/vdf/index.php"><div class="button">Learn More</div></a></center>
          </li>
          <li>
            <img src="img/inDB.png">
            <h3>In-Database Data Science</h3>
            <div class="description">VerticaPy takes advantage of Vertica in-database ML and advanced SQL functionnality, providing plenty of abstractions to simplify the data science process.</div>
            <center><a href="./documentation_last/learn/examples/index.php"><div class="button">Learn More</div></a></center>
          </li>
        </ul></center>
      <h1>Features</h1>
      <p>
        VerticaPy is the perfect blend of the scalability of Vertica and the flexibility of Python, bringing a unique and indispensible set of data science tools.
      </p>
          <br>
          <p>Explore your data.</p>
        <br><center><ul id="advantages">
          <a href="documentation_last/vdataframe/statistics.php"><li>
            <img src="img/icons/Budget.png">
            <h3>Descriptive Statistics</h3>
            <div>Data Exploration</div>
          </li></a>
          <a href="documentation_last/vdataframe/plot.php"><li>
            <img src="img/icons/Pie Chart.png">
            <h3>Charts</h3>
            <div>Data Vizualization</div>
          </li></a>
          <a href="documentation_last/vdataframe/corr.php"><li>
            <img src="img/icons/Co-Working.png">
            <h3>Correlation & Dependancy</h3>
            <div>Statistical Relationships</div>
          </li></a>
        </ul></center><br>
      <p>
        Prepare your Data with advanced Features Engineering using Advanced Analytical Functions and Moving Windows.
      </p>
        <br><center><ul id="advantages">
          <a href="documentation_last/vdataframe/features.php"><li>
            <img src="img/icons/Fresh Idea.png">
            <h3>Features Engineering</h3>
            <div>Advanced Analytical Functions</div>
          </li></a>
          <a href="documentation_last/vdataframe/preprocessing.php"><li>
            <img src="img/icons/SEO Content Management.png">
            <h3>Data Cleaning</h3>
            <div>Smart Preprocessing</div>
          </li></a>
        </ul></center><br>
      <p>
        Create a model with the highly scalable Vertica ML. Effortlessly build and evaluate models that optimize for efficiency and performance using many of the in-database, scalable ML algorithims.
      </p>
      <br><center><ul id="advantages">
          <a href="documentation_last/learn/examples/regression/index.php"><li>
            <img src="img/icons/Analysis.png">
            <h3>Regression</h3>
            <div>Forecast sales revenues.</div>
          </li></a>
          <a href="documentation_last/learn/examples/binary-classification/index.php"><li>
            <img src="img/icons/Teamwork Problem Fix.png">
            <h3>Binary Classification</h3>
            <div>Predict sensor failure.</div>
          </li></a>
          <a href="documentation_last/learn/examples/unsupervised/index.php"><li>
            <img src="img/icons/Conference.png">
            <h3>Clustering</h3>
            <div>Customers segmentation.</div>
          </li></a>
          <a href="documentation_last/learn/examples/multiclass-classification/index.php"><li>
            <img src="img/icons/Business Hierarcy.png">
            <h3>Multiclass Classification</h3>
            <div>Classify gene expression data.</div>
          </li></a>
          <a href="documentation_last/learn/examples/decomposition/index.php"><li>
            <img src="img/icons/Management.png">
            <h3>Decomposition</h3>
            <div>Dimensionality Reduction.</div>
          </li></a>
        </ul></center><br>
      <p>
          This all takes place where it should: <b>your database</b>. By aggregating your data with Vertica, you can build, analyze, and model anything without modifying your data.
      </p>
      <div><?php include('include/footer.php'); ?></div>
      </div>
    </body>
</html>
