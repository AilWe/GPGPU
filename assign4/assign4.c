#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

GLubyte* loadImage(char* filename, int* width, int* height);

static float rotate = 0.0f;
static GLuint texName;
static GLubyte *myTextImage;
static int textWidth, textHeight;

void init(void){
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glShadeModel(GL_SMOOTH);
  GLfloat light_ambient[] = {0.0, 0.0, 1.0, 1.0};
  GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
  GLfloat light_position[] = {0.0, 0.0, 7.0, 0.0};

  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_DEPTH_TEST);

  myTextImage = loadImage("scuff.ppm", &textWidth, &textHeight);
  printf("texture image size: %d %d\n", textWidth, textHeight);

  glGenTextures(1, &texName);
  glBindTexture(GL_TEXTURE_2D, texName);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textWidth, textHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, myTextImage);
  
}

void display(void){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texName);
  //glColor3f(1.0, 1.0, 1.0);
  glPushMatrix();
  glRotatef(rotate, 0.0, 1.0, 0.0);
  GLUquadric *quad = gluNewQuadric();
  gluQuadricTexture(quad, 1);
  gluSphere(quad, 5, 20, 20);
//  glutWireSphere(1.0, 20, 16);
//  glScalef(1.0, 1.0, 1.0);
  glPopMatrix();
  glutSwapBuffers();
  glDisable(GL_TEXTURE_2D);
}

void reshape(int w, int h){
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void update(int value){
  rotate += 2.0f;
  if(rotate > 360.f) rotate -= 360;
  glutPostRedisplay();
  glutTimerFunc(25, update, 0);
}

int main(int argc, char** argv){
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(100, 100);
  glutCreateWindow(argv[0]);
  init();
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutTimerFunc(25, update, 0);
  glutMainLoop();
  return 0;
}

GLubyte* loadImage(char* filename, int* width, int* height){
  FILE *fd;
  int i, w, h, d;
  GLubyte* image;
  char head[70];

  fd = fopen(filename, "rb");
  if (!fd) {
    printf("%s doesn't exist or can't be opened!\n", filename);
    exit(0); 
  }

  fgets(head, 70, fd);
  if (strncmp(head, "P6", 2)){
    printf("%s is not a PPM file!\n", filename);
    exit(0);
  }
  
  i = 0;
  while(i<3){
    fgets(head, 70, fd);
    if(head[0] == '#'){
//      printf("%s\n", head);
      continue;
    }
    if(i==0)
      i += sscanf(head, "%d %d %d", &w, &h, &d);
    else if (i==1)
      i += sscanf(head, "%d %d", &h, &d);
    else if (i==2)
      i += sscanf(head, "%d", &d);
  }
  image = (GLubyte *)malloc(3*sizeof(GLubyte)*w*h);
  fread(image, sizeof(GLubyte), w*h*3, fd);
  fclose(fd);
  
  *width = w;
  *height = h;
  return image;
}


