from django.contrib import admin
from django.urls import path, include
from VSA_website import views

urlpatterns = [
    path('',views.index,name='home'),
    path('about',views.about,name='about'),
    path('about/contributors',views.contributors,name='contributors'),
    path('doubt',views.doubt,name='doubt'),
    path('login',views.login,name='login'),
    path('receive',views.receive,name='login'),
    path('print_page',views.print_page,name='print_page'),
]

