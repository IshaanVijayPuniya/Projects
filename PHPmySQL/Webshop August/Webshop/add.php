<?php
    require('db.php');

    if (isset($_REQUEST['name'])) {

    // When form submitted, insert values into the database.
        // removes backslashes
        $name = ($_REQUEST['name']);
        $name = mysqli_query($con, $name);
        $image    = ($_REQUEST['email']);
        $image    = mysqli_query($con, $image);
        $quantity = ($_REQUEST['quantity']);
        $quantity = mysqli_query($con, $quantity);
        $rrp = ($_REQUEST['rrp']);
        $rrp = mysqli_query($con, $rrp);
        $price = ($_REQUEST['price']);
        $price = mysqli_query($con, $price);
        $id = ($_REQUEST['id']);
        $id = mysqli_query($con, $id);
        $create_datetime = date("Y-m-d H:i:s");
        
        $query    = "INSERT into `products` (`name`, `price`,`rrp`, `quantity`,`img`,`id`,`date_added`)
                     VALUES ('$name', '$price', '$rrp', '$quantity','$image','$id','$create_datetime')";
        $result   = mysqli_query($con, $query);
        if ($result) {
            echo "Product added succesfully ";
        } else {
            echo "product not added";
        }
    } else {
?>
<form class="form" action="" method="post">
        <h1 class="login-title">Add Products</h1>
        <input type="text" class="login-input" name="name" placeholder="name" required />
        <input type="number" class="login-input" name="id" placeholder="ID" required />
        <input type="text" class="login-input" name="email" placeholder="Image" required>
        <input type="number" class="login-input" name="quantity" placeholder="quantity" required>
        <input type="number" class="login-input" name="rrp" placeholder="rrp" required>
        <input type="number" class="login-input" name="price" placeholder="price" required>
        <input type="submit" name="submit" value="Add" class="login-button">
        
    </form>
    
    <?php
    }
   

?>