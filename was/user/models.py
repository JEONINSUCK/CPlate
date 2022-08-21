from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager  #BaseUserManager 추가


# Create your models here.

class UserManager(BaseUserManager):
   def _create_user(self, email, username, password, gender=2, **extra_fields):
      email = self.model.normalize_username(email)
      user = self.model(email=email, **extra_fields)
      user.set_password(password)
      user.save()
      return user

   def create_user(self, email, username='', password=None, **extra_fields):
       ...
       return self._create_user(email, username, password, **extra_fields)

   def create_superuser(self, email, password, **extra_fields):
       ...
       return self._create_user(email, '~~.html',  password, **extra_fields)

class User(AbstractUser):
   email = models.EmailField(verbose_name='email', unique=True)
   username = models.CharField
   gender = models.SmallIntegerField
   objects = UserManager()

   USERNAME_FIELD = 'email'  # email 로 로그인
   REQUIRED_FIELDS = [] # 필수로 받고 싶은 필드들 넣기 원래 소스 코드엔 email필드가 들어가지만 우리는 로그인을 이메일로
   VERIFY_FIELDS = []  # 회원가입 시 검증 받을 필드 (email, phone)
   REGISTER_FIELDS = ['email', 'password']  # 회원가입 시 입력 받을 필드 (email, phone, password)

   def __str__(self):
    return "<{} {}>".format(self.email, self.username)
      