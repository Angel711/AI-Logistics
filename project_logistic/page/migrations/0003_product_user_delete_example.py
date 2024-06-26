# Generated by Django 4.1.13 on 2024-04-20 09:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('page', '0002_example_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('productName', models.CharField(max_length=50)),
                ('productPrice', models.IntegerField()),
                ('productQuantity', models.IntegerField()),
                ('productDescription', models.TextField()),
                ('productImage', models.ImageField(upload_to='product_images/')),
                ('productCategory', models.CharField(max_length=50)),
                ('productStatus', models.CharField(max_length=50)),
                ('productTendency', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userName', models.CharField(max_length=50)),
                ('lastName', models.CharField(max_length=50)),
                ('email', models.EmailField(max_length=254)),
                ('phone', models.CharField(max_length=20)),
                ('password', models.CharField(max_length=50)),
                ('role', models.CharField(max_length=50)),
                ('status', models.CharField(max_length=50)),
            ],
        ),
        migrations.DeleteModel(
            name='Example',
        ),
    ]
