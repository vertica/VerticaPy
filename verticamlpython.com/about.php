<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>Why Vertica ML Python ?</h1>
        <p>
          Nowadays, the 'Big Data' (Tb of data) is one of the main topics in the Data Science World. Data Scientists are very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. 
        </p><br>
        <p>
          Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data.
        </p><br>
        <p>
          Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. The idea behind Vertica ML Python is simple: Combining the Scalability of Vertica with the Flexibility of Python to give to the community what they need: <b>Bringing the logic to the data and not the opposite</b>.
        </p><br>
        <center>
          <img src="img/vertica-ml-python.png" width="80%">
        </center>
        <h1>History</h1>
        <p>
          With the arrival of new technologies which made Data Science Possible, optimization was not the first need. Most of the companies didn't thought about the evolution of Data Storage and Ingestion. This fast growth didn't let place for chance, all the companies needed to change their ways to manage data. DataBases were still considered as Data Warehouse. Many in-memory technologies tried to follow the trend but they never succeeded for high data volumes. Indeed, these technologies need powerful Hardwares and time to move the data which is sometimes impossible due to security.
        </p><br>
        <p>
          By seeing all these facts, Vertica decided to help their customers by bringing the logic to the data. Vertica implemented the first in-DB Machine Learning algorithms built to scale. This was in 2015 and since now, all the DataBases are trying to follow the vision.
        </p><br>
        <p>
          SQL is easy to learn but it has a lack of flexibility due to the absence of loops and variables creation. Python is flexible but is not scalable. By creating the combination of both of the technologies, Vertica ML Python is bringing what Data Scientists need. This idea emerged in 2017 when Badr Ouali was still data scientist intern at Vertica. 
        </p><br>
        <p>
          A first release was made in 2018 in Github. The version was incomplete. Vertica ML Python needed 2 years of development and technologies expertise to become what it is today. 
        </p>
        <h1>First Official Logo</h1>
        <p>You can see following the first draft of the Vertica ML Python Logo.</p><br>
        <center>
          <img src="img/first-logo.png" width="30%" style="min-width: 300px;">
        </center>
        <h1>A Few Words from the Creator</h1>
        <p>"This Python Module is the result of my passion for Data Science. I love discovering everything possible in the data. I always kept a passion for mathematics and specially for statistics. When I saw the lack of libraries using as back-end the power of columnar MPP Database, I decided to help the Data Science Community by bringing the logic to the data."</p><br>
        <center>
          <img src="img/Badr.jpg" width="30%" style="min-width: 300px;">
        </center>
        <div><?php include('include/footer.php'); ?></div>
      </div>
    </body>
</html>