<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Connect</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="available_auto_connection/index.php">available_auto_connection</a></td> <td>Displays all available auto connections.</td> </tr>
          <tr> <td><a href="change_auto_connection/index.php">change_auto_connection</a></td> <td>Changes the current auto connection.</td> </tr>
          <tr> <td><a href="new_auto_connection/index.php">new_auto_connection</a></td> <td>Saves a connection to automatically create DB cursors.</td> </tr>
          <tr> <td><a href="read_auto_connect/index.php">read_auto_connect</a></td> <td>Automatically creates a connection from the connection made by new_auto_connection.</td> </tr>
          <tr> <td><a href="read_dsn">read_dsn</a></td> <td>Reads the DSN information from the ODBCINI environment variable.</td> </tr>
          <tr> <td><a href="to_vertica_python_format/index.php">to_vertica_python_format</a></td> <td>Converts the ODBC dictionary obtained with the read_dsn method to the vertica_python format.</td> </tr>
          <tr> <td><a href="vertica_cursor/index.php">vertica_cursor</a></td> <td>Reads the input DSN from the ODBCINI environment and creates a Vertica database cursor using the input method.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
