
<?php

    require('db.php');
    $sql = "SELECT id, username, email FROM users";
    $result = $con->query($sql);
    echo"";
    if ($result->num_rows > 0) {
      // output data of each row
      while($row = $result->fetch_assoc()) {
        echo " <br/><table  border = '1'> <tr>|| id:   " . $row["id"]. "   </tr> <tr> ||Name: " . $row["username"]. "  </tr><tr>|| Email- " . $row["email"]. "</tr></table>";
      }
    } else {
      echo "0 results";
    }




    // When form submitted, insert values into the database.
    if (isset($_REQUEST['username'])) {
        // removes backslashes
        $username = stripslashes($_REQUEST['username']);
        //escapes special characters in a string
        $username = mysqli_real_escape_string($con, $username);
        $email    = stripslashes($_REQUEST['email']);
        $email    = mysqli_real_escape_string($con, $email);
        $password = stripslashes($_REQUEST['password']);
        $password = mysqli_real_escape_string($con, $password);
        $create_datetime = date("Y-m-d H:i:s");
        $query    = "INSERT into `users` (username, password, email, create_datetime)
                     VALUES ('$username', '" . md5($password) . "', '$email', '$create_datetime')";
        $result   = mysqli_query($con, $query);
        if ($result) {
            echo "User added succesfully ";
        } else {
            echo "User could not be added";
        }
    } else {
?>
<html><link href="style.css" rel="stylesheet">
    <form class="form" action="" method="post">
        <h1 class="login-title">Add User</h1>
        <input type="text" class="login-input" name="username" placeholder="Username" required />
        <input type="text" class="login-input" name="email" placeholder="Email Adress" required>
        <input type="password" class="login-input" name="password" placeholder="Password" required >
        <input type="submit" name="submit" value="Add this user" class="login-button">
        
    </form>
    </html>
<?php
    }
    include('delete.php');
    include('add.php');
    include('update.php');

?>

