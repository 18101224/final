package controller.GameControl;

import controller.block.Block;
public class GameAreaController extends GameArea implements gameFunction{
    private GameArea ga;
    public GameAreaController(){
        this.ga=new GameArea();
    }
    public boolean checkBottom() {
        if (ga.block.getBottomEdge() == ga.gridRows) {
            return false;
        }
        int[][] shape = ga.block.getShape();
        int w = ga.block.getWidth();
        int h = ga.block.getHeight();
        for (int col = 0; col < w; col++) {
            for (int row = h - 1; row >= 0; row--) {
                if (shape[row][col] != 0) {
                    int x = col + ga.block.getX();
                    int y = row + ga.block.getY() + 1;
                    if (y < 0) break;
                    if (ga.background[y][x] != 0) return false;
                    break;
                }
            }
        }
        return true;
    }
    private boolean checkLeft() {
        if (ga.block.getLeftEdge() == 0) return false;
        int[][] shape = ga.block.getShape();
        int w = ga.block.getWidth();
        int h = ga.block.getHeight();
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (shape[row][col] != 0) {
                    int x = col + ga.block.getX() - 1;
                    int y = row + ga.block.getY();
                    if (y < 0) break;
                    if (ga.background[y][x] != 0) return false;
                    break;
                }
            }
        }
        return true;
    }
    private boolean checkRight() {
        if (ga.block.getRightEdge() == ga.gridColumns) return false;
        int[][] shape = ga.block.getShape();
        int w = ga.block.getWidth();
        int h = ga.block.getHeight();
        for (int row = 0; row < h; row++) {
            for (int col = w - 1; col >= 0; col--) {
                if (shape[row][col] != 0) {
                    int x = col + ga.block.getX() - 1;
                    int y = row + ga.block.getY();
                    if (y < 0) break;
                    if (ga.background[y][x] != 0) return false;
                    break;
                }
            }
        }
        return true;
    }
    @Override
    public void spawnBlock(int bln) { //????????????
        ga.block = new Block(bln);
        ga.block.spawn(gridColumns);
    }
    @Override
    public boolean isBlockOuOofBounds() {
        if (ga.block.getY() < 0) { //??? ??? ???????????? ?????????, ???????????? ??? ??????
            ga.block = null;
            return true;
        }
        return false;
    }
    @Override
    public int[][] moveBlockDown() {
        if(checkBottom()==false){return background;}
        ga.block.moveDown();
        return ga.background;
    }
    @Override
    public int[][] moveBlockRight() {
        if (ga.block == null) return ga.background;
        if (!checkRight()) return ga.background;
        ga.block.moveRight();
        return ga.background;
    }
    @Override
    public int[][] moveBlockLeft() {
        if (ga.block == null) return ga.background;
        if (!checkLeft()) return ga.background;
        ga.block.moveLeft();
        return ga.background;
    }
    @Override
    public int[][] dropBlock() {
        if (ga.block == null) return ga.background;
        while (checkBottom()) {
            ga.block.moveDown();
        }
        return ga.background;
    }
    @Override
    public int[][] rotateBlock(){
        if (ga.block == null) return ga.background;
        //if (ga.block.getHeight()+ ga.block.getX() > gridColumns ){ ga.block.moveLeft();}
        ga.block.rotate();
        return ga.background;
    }
    @Override
    public void moveBlockToBackground(){
        int[][] shape = ga.block.getShape();
        int h = ga.block.getHeight();
        int w = ga.block.getWidth();
        int xPos = ga.block.getX();
        int yPos = ga.block.getY();
        int color = ga.block.getColor();
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                if (shape[r][c] != 0) {
                    ga.background[r + yPos][c + xPos] = color;
                }
            }
        }
    }

    @Override
    //?????????
    public int clearLines() {
        boolean lineFilled;
        int linesCleared = 0;
        for (int r = ga.gridRows - 1; r >= 0; r--) {
            lineFilled = true;
            for (int c = 0; c < ga.gridColumns; c++) {
                if (ga.background[r][c] == 0) {
                    lineFilled = false;
                }

            }
            if (lineFilled) {
                linesCleared++;
                clearLine(r);
                shiftDown(r);
                clearLine(0);
                r++; //????????? ??????????????? ??????
            }

        }
        return linesCleared;
    }
    //?????????

    private void clearLine(int r) {

        for (int i = 0; i < ga.gridColumns; i++) {
            ga.background[r][i] = 0;
        }
    }
    //????????? ?????? ????????????
    private void shiftDown(int r) {
        for (int row = r; row > 0; row--) {
            for (int col = 0; col < ga.gridColumns; col++) {
                ga.background[row][col] = ga.background[row - 1][col];
            }
        }
    }
    public int[][] getBackground(){
        return ga.background;
    }
    public int[][] newBackground(){
        int[][] newBackground = ga.getBackground();
        int x=ga.block.getX();
        int y=ga.block.getY();
        for(int r=x; r<x+ga.block.getHeight();r++){
            for(int c=y;c<ga.block.getHeight()+y;c++){
                if(ga.block.getShape()[r-x][c-y]==1){
                    newBackground[r][c]=ga.block.getColor();
                }
            }
        }
        return newBackground;
    }
    public int getY(){
        return ga.block.getY();
    }
}
